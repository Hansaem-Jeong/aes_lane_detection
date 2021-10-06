#!/usr/bin/env python3

# hansaem
import rospy
from sensor_msgs.msg import Image
import ros_numpy 
import threading as thread
import tensorflow as tf


####

import logging
import argparse
import os
import torch

from lib.config import Config
from lib.experiment import Experiment
#from lib.runner import Runner
import yaml
from natsort import natsorted
import re
import numpy as np
import cv2


import pickle
import random

from tqdm import tqdm, trange

from cv_bridge import CvBridge, CvBridgeError
import time
from time import sleep
import torch.nn
from torch.autograd import Variable
import torchvision
from torchvision import transforms
import PIL

import sys
#sys.path.append("/home/avees/hansaem/lane_ws/src/lane_detection/")
#sys.path.append("/home/avees/hansaem/lane_ws/src/lane_detection/scripts")


class Runner:
    def __init__(self, cfg, exp, device, test_dataset,test_first_dir,test_second_dir,exp_name,hyper,hyper_param,video_name,root_path,resume=False, view=None, deterministic=False):
        self.cfg = cfg
        self.exp = exp
        self.device = device
        self.resume = resume
        self.view = view
        self.test_dataset=test_dataset
        self.test_first_dir=test_first_dir
        self.test_second_dir=test_second_dir
        self.logger = logging.getLogger(__name__)

        print("4444444444444444444444444444444444444444444444444444")
        print(self.test_dataset)
        self.dataset_type=hyper_param[3]
        self.conf_threshold=hyper_param[0]
        self.nms_thres=hyper_param[1]
        self.nms_topk=hyper_param[2]

        self.root=root_path
        self.video_name=video_name
        self.hyper=hyper
        print(self.root)
        self.exp_name = "/{}/{}/".format(exp_name,self.hyper)
        self.name = test_first_dir + test_second_dir +test_dataset
        print(self.name)
        self.log_dir = self.name+self.exp_name#os.path.join(self.name,self.exp_name)
        print(self.log_dir)
        os.makedirs(self.log_dir, exist_ok=True)

        # Fix seeds
        torch.manual_seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        random.seed(cfg['seed'])

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.lane_detection_init()

    def lane_detection_init(self):
        self.lock = thread.Lock()
        self.camera_topic = rospy.get_param("~camera_topic")

        sub_camera = rospy.Subscriber(self.camera_topic, Image, self.laneCb, queue_size=1)

        self.bridge = CvBridge()

        self.camera_status = False
 
        self.data_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(360, 640)),
            torchvision.transforms.ToTensor()])

        self.max_iter = 150
        self.count = 0 # loop counter
        self.cycle_time = np.zeros(self.max_iter)

    def laneCb(self, data):
        try:
            lane_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            print(e)
                      
# lock  
        self.lock.acquire()

        self.recv_timestamp = time.monotonic()*1000.
        
       
        #self.camera_image = lane_image
# unlock
        self.lock.release()

        self.camera_image = lane_image
        self.camera_status = True


    def train(self):
        self.exp.train_start_callback(self.cfg)
        starting_epoch = 1
        model = self.cfg.get_model()
        model = model.to(self.device)
        optimizer = self.cfg.get_optimizer(model.parameters())
        scheduler = self.cfg.get_lr_scheduler(optimizer)
        if self.resume:
            last_epoch, model, optimizer, scheduler = self.exp.load_last_train_state(model, optimizer, scheduler)
            starting_epoch = last_epoch + 1
        max_epochs = self.cfg['epochs']
        train_loader = self.get_train_dataloader()
        loss_parameters = self.cfg.get_loss_parameters()
        for epoch in trange(starting_epoch, max_epochs + 1, initial=starting_epoch - 1, total=max_epochs):
            self.exp.epoch_start_callback(epoch, max_epochs)
            model.eval()
            pbar = tqdm(train_loader)
            for i, (images, labels, _) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = model(images, **self.cfg.get_train_parameters())
                loss, loss_dict_i = model.loss(outputs, labels, **loss_parameters)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Scheduler step (iteration based)
                scheduler.step()

                # Log
                postfix_dict = {key: float(value) for key, value in loss_dict_i.items()}
                postfix_dict['lr'] = optimizer.param_groups[0]["lr"]
                self.exp.iter_end_callback(epoch, max_epochs, i, len(train_loader), loss.item(), postfix_dict)
                postfix_dict['loss'] = loss.item()
                pbar.set_postfix(ordered_dict=postfix_dict)
            self.exp.epoch_end_callback(epoch, max_epochs, model, optimizer, scheduler)

            # Validate
            if (epoch + 1) % self.cfg['val_every'] == 0:
                self.eval(epoch, on_val=True)
        self.exp.train_end_callback()


    def numpy_to_torch(self, image):
        image = PIL.Image.fromarray(image)
        image = self.data_transforms(image)
        return image


    def preprocess(self, image):
        preprocess_start = time.time()
        self.lock.acquire()
        image = PIL.Image.fromarray(image)

#        image = torch.from_numpy(image).float()
#        print("hansaem preprocess shape :", image.shape)
#        image = image.reshape(360,640,3)
        image = self.data_transforms(image)
#        print("hansaem image shape1:", image.shape)
        image = image.permute(0,1,2)
#        print("hansaem image shape2:", image.shape)
        image = image.float()
#        print("hansaem image shape3:", image)
#        image = image.cuda()
        image = image.unsqueeze(0)
        
        self.lock.release()
        preprocess_end = time.time()
        self.preprocess_time = preprocess_end - preprocess_start
        self.image_infer = image
#        print("hansaem image shape4:", image.shape)

    def eval(self, epoch, on_val=False, save_predictions=False):
        #prediction_name="predictions_r34_culane"#
        model = self.cfg.get_model()
        model_path = self.exp.get_checkpoint_path(epoch)
#        print("\nhansaem eval ", model_path, "\n", model)
        self.logger.info('Loading model %s', model_path)
        model.load_state_dict(self.exp.get_epoch_model(epoch))
        model = model.to(self.device)
        model.eval()
        if on_val and self.test_dataset ==None:
            dataloader = self.get_val_dataloader()
#            print("1111111111111111111111111111111111111111111111\n")
        elif self.test_dataset !=None:
            dataloader =self.get_kodas_test_dataloader()
#            print("2222222222222222222222222222222222222222222222\n")
#            print(self.test_dataset)
        else:
            dataloader = self.get_test_dataloader()
#            print("3333333333333333333333333333333333333333333333\n")
        test_parameters = self.cfg.get_test_parameters()
#        print("hansaem ", dataloader)
        predictions = []
        self.exp.eval_start_callback(self.cfg)

        while not self.camera_status:
            print("Waiting for {0:s} camera data...".format(self.camera_topic))
            sleep(2)

        print("hansaem check 1")
### Multi-
#        self.lock.acquire()
#        images_p = self.numpy_to_torch(self.camera_image)
#        self.lock.release()
#        self.input_image = images_p

        self.preprocess(self.camera_image)
        print("hansaem check 2")
#        self.preprocess(self.input_image)
        self.inference_image(self.image_infer, model, test_parameters)
        print("hansaem check 3")
        self.preprocess(self.camera_image)

#        self.lock.acquire()
#        images_p = self.numpy_to_torch(self.camera_image)
#        self.lock.release()
#        self.input_image = images_p

 #       self.preprocess(self.input_image)
####        

#        with torch.no_grad():
##            for idx, (images, _, _) in enumerate(tqdm(dataloader)):
#            for idx, (images, _, _) in enumerate(dataloader):
##hansaem
#                start_times = time.time()
#                images = images.to(self.device)
#                output = model(images, **test_parameters)
#                prediction = model.decode(output, as_lanes=True)
##                print("\nhansaem prediction pre- : ", prediction[0])
#                predictions.extend(prediction)
##                print("\nhansaem prediction post- : ", prediction)
#                if self.view:
#                    img = (images[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
#                    img, fp, fn = dataloader.dataset.draw_annotation(idx, img=img, pred=prediction[0])
#                    if self.view == 'mistakes' and fp == 0 and fn == 0:
#                         continue
#
#                    __name=self.log_dir+str(idx) + '.jpg'
#
##                    cv2.imwrite(__name, img)
##                img = (images[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
#                    cv2.imshow("KODAS", img)
#                    cv2.waitKey(1)
##hansaem
#                cycle_times = time.time() - start_times
#                print(f"{idx:d} iter\n{cycle_times:.5f} sec")

#        p = transforms.Compose([transforms.Resize((360, 640))])
        while not rospy.is_shutdown():
            print("===============")
            start_times = time.time()
#lock
#            image = cv2.resize(self.lane_image, dsize=(640,360),interpolation=cv2.INTER_AREA)
#            image = torch.Tensor(tuple(image))
#            before_preprocess = time.time()
### Multi-
#            preprocess_thread = thread.Thread(target = self.preprocess, args=(self.input_image,))
            preprocess_thread = thread.Thread(target = self.preprocess, args=(self.camera_image,))
            inference_thread = thread.Thread(target = self.inference_image, args=(self.image_infer, model, test_parameters,))

            preprocess_thread.start()
            inference_thread.start()

            display_start = time.time()
#            self.lock.acquire()
#            images_p = self.numpy_to_torch(self.camera_image)
#            self.lock.release()
#            self.input_image = images_p

####
            

#            print("hansaem image shape1:", image.shape)
#            image = image.permute(2,0,1)
#            print("hansaem image shape2:", image.shape)
#            image = image.unsqueeze(0)
#            print("hansaem image shape3:", image)

### Sequential
#            self.preprocess(self.camera_image)
#            self.inference_image(self.image_infer, model, test_parameters)
####
#            self.lock.acquire()
                       
            if self.view:
                img = (self.image_disp[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                after_cpu = time.time()
                img, fp, fn = dataloader.dataset.draw_annotation(self.count, img=img, pred=self.prediction[0])
                after_draw = time.time()
                if self.view == 'mistakes' and fp == 0 and fn == 0:
                    continue
        
                __name=self.log_dir+str(self.count) + '.jpg'
        
                cv2.imshow("KODAS", img)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

#            self.lock.release()
### Multi-
            display_end = time.time()
            preprocess_thread.join()
            inference_thread.join()
####

            after_cycle = time.time()
            cycle_times = after_cycle - start_times
            self.display_time = display_end - display_start
            
            print("      Cycle time :", cycle_times, "ms")
            print("  Inference time :", self.inference_time, "ms")
            print(" Preprocess time :", self.preprocess_time, "ms")
            print("    Display time :", self.display_time, "ms")

            self.count += 1
            if self.count == 800:
                break



        print("hansaem hansaem hansaem\n")
        image_folder = self.log_dir
        video_name =  self.log_dir+self.video_name+'.avi'
        images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
        images = natsorted(images)
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, 0, 30, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()

        # if save_predictions:
        #     with open("/data2/lane_data/LaneATT/prediction/8_7/"+prediction_name+'.pkl', 'wb') as handle:
        #         pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # self.exp.eval_end_callback(dataloader.dataset.dataset, predictions, epoch)

#hansaem
#    def ros_run(self):

    def inference_image(self, image, model, test_parameters): 
        inference_start = time.time()
        image = image.to(self.device)
#            image = tf.image.decode_image(image, channels=3)
#            image = tf.image.convert_image_dtype(image, tf.float32)
              
#            image = np.expand_dims(self.lane_image, axis=0)
#            images = torch.from_numpy(list(image))
#            images = torch.expand_dims(image, axis=0)
#            images = tf.tensor4d(Array.from(resized.dataSync()), [64, 3, 7, 7])
#unlock

        output = model(image, **test_parameters)
        self.prediction = model.decode(output, as_lanes=True)
        inference_end = time.time()
        self.inference_time = inference_end - inference_start
        self.image_disp = image

                    
#hansaem
           
   
    def get_train_dataloader(self):
        train_dataset = self.cfg.get_dataset('train')
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=self.cfg['batch_size'],
                                                   shuffle=True,
                                                   num_workers=8,
                                                   worker_init_fn=self._worker_init_fn_)
        return train_loader

    def get_kodas_test_dataloader(self):
        self.cfg.set_kodas('test', self.dataset_type,self.conf_threshold, self.nms_thres, self.nms_topk,self.root)
        test_dataset = self.cfg.get_dataset('test')
        print("55555555555555555555555555555555555555555555")
        print(test_dataset)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=self.cfg['batch_size'] if not self.view else 1,
                                                  shuffle=False,
                                                  num_workers=8,
                                                  worker_init_fn=self._worker_init_fn_)
        return test_loader

    def get_test_dataloader(self):
        test_dataset = self.cfg.get_dataset('test')
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=self.cfg['batch_size'] if not self.view else 1,
                                                  shuffle=False,
                                                  num_workers=8,
                                                  worker_init_fn=self._worker_init_fn_)
        return test_loader

    def get_val_dataloader(self):
        val_dataset = self.cfg.get_dataset('val')
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=self.cfg['batch_size'],
                                                 shuffle=False,
                                                 num_workers=8,
                                                 worker_init_fn=self._worker_init_fn_)
        return val_loader

    @staticmethod
    def _worker_init_fn_(_):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2**32 - 1
        random.seed(torch_seed)
        np.random.seed(np_seed)


   


class LaneDetection:
    def __init__(self): #parse_args():
        rospy.init_node('lane_detection_node', anonymous = True)
    
        namesp = argparse.Namespace()
    
        setattr(namesp, 'mode', rospy.get_param("mode", 'test'))
        setattr(namesp, 'exp_name', rospy.get_param("~exp_name"))
        setattr(namesp, 'cfg', rospy.get_param("cfg", None))
        setattr(namesp, 'resume', rospy.get_param("resume", False))
        setattr(namesp, 'epoch', rospy.get_param("epoch", None))
        setattr(namesp, 'cpu', rospy.get_param("cpu", False))
        setattr(namesp, 'save_predictions', rospy.get_param("save_predictions", False))
        setattr(namesp, 'view', rospy.get_param("view", all))
        setattr(namesp, 'test_first_dir', rospy.get_param("~test_first_dir"))
        setattr(namesp, 'test_second_dir', rospy.get_param("~test_second_dir"))
        setattr(namesp, 'test_dataset', rospy.get_param("~test_dataset"))
        setattr(namesp, 'video_name', rospy.get_param("~video_name"))
        setattr(namesp, 'conf_threshold', rospy.get_param("conf_threshold", 0.4))
        setattr(namesp, 'nms_thres', rospy.get_param("nms_thres", 45.))
        setattr(namesp, 'max_lane', rospy.get_param("max_lane", 2))
        setattr(namesp, 'data_dir', rospy.get_param("~data_dir"))
        setattr(namesp, 'deterministic', rospy.get_param("deterministic", False))
        setattr(namesp, 'exps_basedir', rospy.get_param("~exps_basedir"))
        setattr(namesp, 'package_path', rospy.get_param("~package_path"))
    
        #args = parser.parse_args()
        args = namesp
    
        if args.cfg is None and args.mode == "train":
            raise Exception("If you are training, you have to set a config file using --cfg /path/to/your/config.yaml")
        if args.resume and args.mode == "test":
            raise Exception("args.resume is set on `test` mode: can't resume testing")
        if args.epoch is not None and args.mode == 'train':
            raise Exception("The `epoch` parameter should not be set when training")
        if args.view is not None and args.mode != "test":
            raise Exception('Visualization is only available during evaluation')
        if args.cpu:
            raise Exception("CPU training/testing is not supported: the NMS procedure is only implemented for CUDA")
        self.args = args
    
        #return args

    


    def main(self):
        args=self.args
        #os.environ["CUDA_VISIBLE_DEVICES"] = '1'
        hyper="conf_threshold_{}_nms_thres_{}_max_lane_{}".format(args.conf_threshold,args.nms_thres,args.max_lane)
        print("{}_{}".format(args.exp_name,hyper))
        hyper_param = [args.conf_threshold, args.nms_thres, args.max_lane, args.test_dataset]
        exp = Experiment(args.exp_name, args, mode=args.mode, exps_basedir=args.exps_basedir)
        if args.cfg is None:
            cfg_path = exp.cfg_path
        else:
            cfg_path = args.cfg
        args.video_name=hyper
        cfg = Config(cfg_path)
        exp.set_cfg(cfg, override=False)
#        device = torch.device('cpu') if not torch.cuda.is_available() or args.cpu else torch.device('cuda')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        runner = Runner(cfg, exp, device,args.test_dataset,args.test_first_dir,args.test_second_dir,args.exp_name,hyper,hyper_param, args.video_name,args.data_dir, view=args.view, resume=args.resume, deterministic=args.deterministic)
        if args.mode == 'train':
            try:
                runner.train()
            except KeyboardInterrupt:
                logging.info('Training interrupted.')
        runner.eval(epoch=args.epoch or exp.get_last_checkpoint_epoch(), save_predictions=args.save_predictions)
        #runner._eval(epoch=args.epoch or exp.get_last_checkpoint_epoch(), save_predictions=args.save_predictions)
    

if __name__ == '__main__':
    #main()
    lane_detection = LaneDetection()
    lane_detection.main()
