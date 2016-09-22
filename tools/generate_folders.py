# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 18:40:51 2014
print cv2.__version__

@author: carlos
"""
import os
import imtools as imt
import shutil
import numpy as np
from actor import Actor
import cv2
from scipy.fftpack.realtransforms import dst


def generate_pose_folders(data = "../data/preprocess/"):
    # Actors
    actors = ["carlos", "chris", "victor", "lakshman", "norma"]
    
    # 2-D dictionary that contains the scene information
    scenes = ["ideal", "medium", "dark", "blanket", "blanketpillow", "pillow"]
    
    # poses or states: (bgn) + 10 poses
    poses  = ["bgn", "soldierU", "fallerU", "soldierD", "fallerD",
                     "logR", "yearnerR", "fetalR", "logL", "yearnerL", "fetalL"]
    
    # camera views
    views  = ["top", "right", "head"]
    
    
    scenario={
            "rgb":
                {"SCENE": scenes, 
                 "VIEW" : views,
                 "POSE" : poses},
                     
            "depth":
                {"SCENE": scenes,
                 "VIEW" : views,
                 "POSE" : poses},
                    
            "mask":
                {"SCENE": scenes,
                 "VIEW" : views,
                 "POSE" : poses},
 
            "pressure":
                {"SCENE": scenes,
                 "VIEW" : ["bottom"],
                 "POSE" : poses},
               
              "binary":
                  {"SCENE": scenes,
                   "VIEW" : views,
                   "POSE" : poses}
             }


    for actor in actors:
        for mode in scenario.keys():
            print "mode: ", mode
            k = scenario[mode].keys()
            print "\t {} contains {} ".format(k[0], scenario[mode][k[0]])
            print "\t {} contains {} ".format(k[1], scenario[mode][k[1]])
            print "\t {} contains {} ".format(k[2], scenario[mode][k[2]])
            
            for scene in scenario[mode]["SCENE"]:
                for view in scenario[mode]["VIEW"]:
                    for pose in scenario[mode]["POSE"]:
                        
#                         if mode == 'binary':
#                             p = data + ('/').join([actor, mode, scene, view])
#                         else:
                        p = data + ('/').join([actor, mode, scene, view, pose])
                        print p
                        imt.generate_folder(p)
#generate_pose_folders()



def copy2organized_folders():
    """From the raw folders -> organized folders using the annotations in Actor.py """
    names = ['carlos']#, 'chris', 'lakshman', 'norma', 'victor'] #'carlos',
    for name in names: #os.listdir("../data/sleep_data"):             # labels
        a = Actor(name)
        # WINDOWS PATHS:
        #p = "../data/sleep_data"
        pp = "../data/preprocess"

        #viewname = "top"
        for viewname in ["right","head", "top"]: #["top", "right", "head"]:
            newp = ('/').join([pp,a.name+'_src',viewname])
            print "=== Path for view: ", newp
            imlist, idxs = imt.get_nat_list(p, name="rgb_", ext = ".png")
      
            idxs.sort()
        
            for scene in a.scenes.keys():
                #scene = 'ideal'
                for pose in a.scenes[scene].keys():
                    #print pose
                    fi, fo = a.scenes[scene][pose]
                    if viewname == "right" and a.name == "norma":
                        fi = fi-10
                        fo = fo-10
                    
                    for frame in range(fi,fo+1):
                        rgb_src   = imlist[np.where(idxs==frame)[0][0]]
                        depth_src = rgb_src.replace("rgb", "depth")
#                        mask_src  = rgb_src.replace("rgb", "mask")
                    
                        rgb_dst   = ("/").join([pp,a.name, "rgb", scene, viewname, pose])
                        depth_dst = rgb_dst.replace("rgb", "depth")
#                        mask_dst  = rgb_dst.replace("rgb", "mask")

                        print "srcfile: ", rgb_src
                        print "dstpath: ", rgb_dst
                        shutil.copy(rgb_src,   rgb_dst)
#                        shutil.copy(mask_src,  mask_dst)
                        shutil.copy(depth_src, depth_dst)
#copy2organized_folders()
    


def copy_all_images_to_single_folder():
    """Copies images in organized folders to a single location with descriptive names."""
    # Paths
    srcpp = "../data/preprocess" # source root
    dstpp = "../data/dataset" # destination
    
    # Actors
    actors = ["chris", "lakshman", "norma", "victor"] #"carlos"
    
    # poses or states: (bgn) + 10 poses
    poses  = ["bgn", "soldierU", "fallerU", "fetalR", "logR", "yearnerR", 
                     "soldierD", "fallerD", "fetalL", "logL", "yearnerL" ]
    # scenes: 6 scene conditions
    scenes = ["blanket", "blanketpillow", "dark", "ideal","medium", "pillow"]
    
    # Camera views
    views  = ["head","right", "top"]
    
    
    scenario={
              "rgb":
                  {"SCENE": scenes, 
                   "VIEW" : views,
                   "POSE" : poses
                  },
                    
            "depth":
                {"SCENE": scenes,
                 "VIEW" : views,
                 "POSE" : poses
                },
                    
            "pressure":
                {"SCENE": scenes,
                 "VIEW" : ["bottom"],
                 "POSE" : poses
                },
               
            "binary":
                {"SCENE": scenes,
                 "VIEW" : views,
                 "POSE" : poses
                }
             }
    count = 0
    for actor in actors:
        for mode in scenario.keys():
            for scene in scenario[mode]["SCENE"]:
                for view in scenario[mode]["VIEW"]:
                    for pose in scenario[mode]["POSE"]:

                        p = ("/").join([srcpp,actor,mode,scene,view, pose])
                        name = ("_").join([actor, pose, scene, view, mode, ""])
                        imnames, idxs = imt.get_nat_list(p, name=name, ext = ".png")
                        idxs.sort()
                        filenames = imt.get_imlist(p, ext = ".png")
#                         if not len(idxs) == len(filenames):
#                             print "\tMismatch in {}".format(p)

#                         print "{} images in {}".format(len(idxs), p)
                        
                        # ==============================
                        # === ONLY COPY TEN IMAGES!! ===
                        # ==============================
                        if len(idxs) < 10:
                            print len(idxs)
                            break
                        
                        for imname in imnames[:10]:
                            count +=1
                            ## copy all files to single location
                            filename = imname.split('\\')[-1]
                            dst = ("/").join([dstpp,filename])
                            shutil.copy(imname,   dst)
                            print "\n src:{} \n dst:{}".format(imname, dst)
        print actor, count
        count =0
#copy_all_images_to_single_folder()



def manicure_data():
    # Paths
    srcpp    = "../data/preprocess"
#     newpp = "../data/mm_sleeping_poses_dataset"
    # Actors
    actors = ["chris"]#"victor", "lakshman", "norma", "carlos", "chris"]
    # poses or states: (bgn) + 10 poses
    poses  = ["bgn", "soldierU", "fallerU", "soldierD", "fallerD", "logR", 
              "yearnerR", "fetalR", "logL", "yearnerL", "fetalL"]
    # scenes: 6 scene conditions
    scenes = ["ideal","medium", "dark", "blanket", "blanketpillow", "pillow"]
    # Camera views
    views  = ["top","right", "head"]
    scenario={
              "rgb":
                  {"SCENE": scenes, 
                   "VIEW" : views,
                   "POSE" : poses
                  },
                   
              "depth":
                  {"SCENE": scenes,
                   "VIEW" : views,
                   "POSE" : poses
                  },
                   
              "pressure":
                  {"SCENE": scenes,
                   "VIEW" : ["bottom"],
                   "POSE" : poses
                  },
              
              "binary":
                  {"SCENE": scenes,
                   "VIEW" : views,
                   "POSE" : poses
                  }
             }
    
    for actor in actors:
        for mode in scenario.keys():
            for scene in scenario[mode]["SCENE"]:
                for view in scenario[mode]["VIEW"]:
                    for pose in scenario[mode]["POSE"]:
                        p = ("/").join([srcpp,actor,mode,scene,view, pose])
                        print "path: ", p
                        
                        name = ("_").join([actor, pose, scene, view, mode, ""])
                        imnames, idxs = imt.get_nat_list(p, name=name, ext = ".png")
                        idxs.sort()
                        if mode == "binary":
                            for ii in range(10):
                                src = imnames[0]
                                dst = src.replace('0',str(ii))
                                print "\tsrc:{} dst:{}".format(src.split('//')[-1], dst.split('//')[-1])
                                if not src == dst:
                                    shutil.copy(src,dst)

                        if mode == "pressure":
                            print'\n'
                            for i in idxs:
                                src = imnames[i]
                                dst = src.replace(str(i),str(i+5))
                                print "\tsrc:{} dst:{}".format(src.split('//')[-1], dst.split('//')[-1])
                                shutil.copy(src,dst)
##manicure_data()


def replicate_and_rename_binary_images():
    
    mode = "binary"
    pp = "../data/preprocess"
    actors = ["carlos", "chris", "victor", "lakshman", "norma"]
    srcscene = 'ideal'
    poses  = ["bgn", "soldierU", "fallerU", "soldierD", "fallerD",
                     "logR", "yearnerR", "fetalR", "logL", "yearnerL", "fetalL"]
    views  = ["top", "right", "head"]


    B = {'ideal': ["medium","dark", "blanket"],
         'pillow': ["blanketpillow"]
        }


    for actor in actors:
        for srcscene in B.keys():
            for dstscene in B[srcscene]:
                for view in views:
                    for pose in poses:
                        srcpath = ("/").join([pp,actor,mode,srcscene,view])
                        
                        print "path: ", srcpath
                        imlist = imt.get_imlist(srcpath, ext='.png')
                        if len(imlist)>0:
                            for src in imlist:
                                dst = src.replace(srcscene, dstscene)
                                print dst
                                shutil.copy(src,  dst)
#duplicate_and_rename_binary_images()


def replicate_and_rename_pressure_images():
    
    mode = "pressure"
    pp = "../data/preprocess"
    
    actors = ["carlos", "chris", "victor", "lakshman", "norma"]
        
    poses  = ["bgn", "soldierU", "fallerU", "soldierD", "fallerD",
                     "logR", "yearnerR", "fetalR", "logL", "yearnerL", "fetalL"]
    views  = ["bottom"]#, "right", "head"]    
    
    
    B = {'ideal': ["medium","dark", "blanket"],
         'pillow': ["blanketpillow"]
        }
    
    for actor in actors[:1]:
        for srcscene in B.keys():
            for dstscene in B[srcscene]:
                for view in views:
                    for pose in poses [1:2]:
                        srcpath = ("/").join([pp,actor,mode,srcscene,view, pose])
                        print "path: ", srcpath
                        imlist = imt.get_imlist(srcpath, ext='.png')
                        if len(imlist)>0:
                            for src in imlist:
                                dst = src.replace(srcscene, dstscene)
                                print dst
                                shutil.copy(src,  dst)
#duplicate_and_rename_pressure_images()


def replicate_and_rename_rgbd_images():
    
    modes = ["rgb", "depth"]
    pp = "../data/preprocess"
    
    actors = ["victor"]#, "carlos","chris", "lakshman", "norma", "victor"]
    scenes = ["ideal", "medium", "dark", "blanket", "blanketpillow", "pillow"]
    views  = ["head", "right", "top"]    
    poses  = ["bgn", "soldierU", "fallerU", "fetalR", "logR", "yearnerR", 
                     "soldierD", "fallerD", "fetalL", "logL", "yearnerL" ]

    for actor in actors:
        for mode in modes:
            for scene in scenes:
                for view in views:
                    for pose in poses:
                        srcp = ("/").join([pp,actor,mode,scene,view,pose])
                        name = ("_").join([actor, pose, scene, view, mode, ""])
                        imnames, idxs = imt.get_nat_list(srcp, name=name, ext = ".png")
                        idxs.sort()
                        if len(idxs)<10:
                            print "{} images in path {}".format(len(idxs), srcp)
                            print "\tReplicate images to complete!"
                            for i in range(len(idxs), 10):
                                dst = imnames[0].replace('0', str(i))
                                print '\tdst: ',dst
                                shutil.copy(imnames[0],  dst)
#duplicate_and_rename_pressure_images()


def resize_and_copy_images():
    """Resize all images and copy to new location """
    print "Resize and Copy Images"
    pp = "../data/mm_sleeping_poses_dataset"
    newpp = "../data/framed"
    imnames = imt.get_imlist(pp, ext = ".png")
        
    for imname in imnames:
        ## === OPERATORS ===
        ## copy all files to single location
        filename = imname.split('\\')[-1] #.split('.')[0]
            
            
        #filename = filename+'_resized.png'
        im = cv2.imread(imname, -1)

        
        dst = ("/").join([newpp,filename])
        cv2.imwrite(dst, r_im)
        
#         shutil.copy(imname,   dst)
        print "\n src:{} \n dst:{} \n name: {}".format(imname, dst,filename)
#resize_and_copy_images()


def frameImage(im, s=320):
    r,c = im.shape[:2]
    
    if r < s: 
        im = cv2.resize(im, (c,s))
    
    ri = int(np.ceil((s-r)/2))
    ro = ri + r
    
    ci = int(np.ceil((s-c)/2))
    co = ci + c
    #print "r,c", r,c
    #print "ri, ro", len(range(ri, ro))
    #print "ci, co", len(range(ci, co))

    if len(im.shape) ==3:
        frame = np.zeros((s,s,3), dtype = im.dtype)
#         print '\t image dims', im.shape
        if r<c:
            frame[ri:ro, :, :] = im  
        else:
            frame[:,ci:co,  :] = im        

    elif len(im.shape) ==2:
        frame = np.zeros((s,s), dtype = im.dtype)
        if r<c:
            frame[ri:ro,:] = im  
        else:
            frame[:,ci:co] = im

    else:
        raise ValueError ( "Invalid image type!")

    return frame
#frameImage(im)
    

def resize_rotate_and_frame():
    print "Rotate and Frame Images"
    srcpp = "../data/dataset"
    dstpp = "../data/framed"
    imnames = imt.get_imlist(srcpp, ext = ".png")
    s=320
    i=0
    for imname in imnames:
        filename = imname.split('\\')[-1]
        ## === OPERATORS ===
        if 'dept' in imname:
            im = cv2.imread(imname, -1)
        else:
            im = cv2.imread(imname)

        b = imt.shrinkIm(im, s)
        
        #rotate the image:
        if 'right' in imname:
            b = np.fliplr(cv2.transpose(b))            
        elif 'head' in imname:
            b = cv2.flip(b, -1)
        #frame the image
        a = imt.frameImage(b)

        dst = ("/").join([dstpp,filename])
        print i, dst, b.shape
        i+=1
        cv2.imshow('rotated', a)            
        cv2.waitKey(5)
                
#         cv2.imwrite(dst, a)
#rotate_and_frame()

def correct_rotation_of_right_view(dstpp="../data/framed"):
    print "Correct Rotation of Right View"
    imnames = imt.get_imlist(dstpp, ext = ".png")
    s=320
    i=0
    for imname in imnames:
        filename = imname.split('\\')[-1]
        if 'right' in filename: 
            ## === OPERATORS ===
            if 'dept' in imname:
                im = cv2.imread(imname, -1)
            else:
                im = cv2.imread(imname)           
            #rotate the image:
            b = np.fliplr(im)            

            print i, imname, b.shape
            i+=1
            cv2.imwrite(imname, b)
            cv2.imshow('rotated', b)
            cv2.waitKey(5)
    # correct_rotation_of_right_view    


def add_tag_to_filename(tag = "_framed.png"):
    newpp = "../data/framed"
    imnames = imt.get_imlist(pp, ext = ".png")
    s=320
    i=0
    for src in imnames:
        dst = src.split('.')[0]
        dst = dst+tag
        os.rename(src, dst)
 #add_framed_to_filename()   


def binarize_images_under_folder():
    root = "../data/framed"
    names = imt.get_imlist(root,ext ='.png')
    for name in names:
        if "binary" in name:
            im = cv2.imread(name)
            if len(np.unique(im)) > 2:
                print "image: {} had {} values".format(name, len(np.unique(im) ))
                im[im>0]=255
                cv2.imwrite(name,im)



def inverse_pressure_images():
    """
    Invert the values of the pressure images to benerate binary masks
    """
    root = "../data/framed"
    names = imt.get_imlist(root,ext ='.png')
    for name in names:
        if "pressure" in name:
#             print name.split('_')[-5]
            im = cv2.cvtColor(cv2.imread(name), cv2.COLOR_RGB2GRAY)
            mask = np.zeros(im.shape, dtype=im.dtype)
            cv2.imshow('input', im)
            crop,_ = imt.cropMask(im, None, th=128)
            mask = np.zeros(crop.shape, dtype=crop.dtype)
            mask[crop>0]=1
            mask[mask!=0]=0
            mask = np.invert(mask)
            #diff[diff!=1]=0
            #diff =diff*255
            crop = cv2.medianBlur(crop,3)
            
            output = mask*crop
            output = frameImage(output)
#             var = mask
#             print "mask: ",type(var), var.dtype, len(np.unique(var)), var.min(), var.max()
#             cv2.imshow('mask', mask*255)
            cv2.imwrite(name,output)

#             var = output
#             print "output: ",type(var), var.dtype, len(np.unique(var)), var.min(), var.max()
            cv2.imshow('output', output)
            
            cv2.waitKey(5)
        #             crop = imt.cropMask(im)
        #             cv2.imshow('crop',im)
        #             cv2.waitKey(10)
#inverse_pressure_images#

def adjust_dataset_names():
    """
    Use this script to adjust the names of the imaegs in teh dataset. 
    old format: <actor>_<pose>_<scene>_<view>_<modality>_<idx>.png
    ** change** : <scene> :: <light>_<condition>
    new format: <actor>_<pose>_<light>_<condition>_<view>_<modality>_<idx>.png
    """
    
    root = "../data/framed"
    imnames = imt.get_imlist(root,ext ='.png')
    for src in imnames:
        dst= src
        if "ideal" in src:
            dst = src.replace("ideal", "bright_clear")
        elif "medium" in src:
            dst = src.replace("medium", "medium_clear")
        elif "dark" in src:
            dst = src.replace("dark", "dark_clear")

        elif "blanket" in src:
            dst = src.replace("blanket", "medium_blanket")
        elif "pillow" in src: 
            dst = src.replace("pillow", "medium_pillow")
        elif "blanketpillow" in src:
            dst = src.replace("blanketpillow", "medium_blanketpillow")
            
#         if "carlos" in src:
#             print "Old name: {} \t || New adjusted name: {}".format(src, dst)
        print "Old name: {} \t || New adjusted name: {}".format(src, dst)            
        os.rename(src, dst)
 #add_framed_to_filename()




def generate_synthetic_dark_bright_rgb_scenes():
    """
    Using the illumination difference between: 
        medium and bright: m2b
        medium and dark: m2d
    it generates new RGB scenes M2B and M2B. 
    """
    view = "head"
    root = "../data/framed"
    
    bri = cv2.imread("../data/framed/lakshman_bgn_bright_clear_"+view+"_rgb_0.png")
    lab_bri = cv2.cvtColor(bri, cv2.cv.CV_RGB2Lab)#.astype(float)
    
    med = cv2.imread("../data/framed/lakshman_bgn_medium_clear_"+view+"_rgb_0.png")
    lab_med = cv2.cvtColor(med, cv2.cv.CV_RGB2Lab)#.astype(float)
    
    drk = cv2.imread("../data/framed/lakshman_bgn_dark_clear_"+view+"_rgb_0.png")
    lab_drk = cv2.cvtColor(drk, cv2.cv.CV_RGB2Lab)#.astype(float)
    
#     cv2.imshow("RGB inputs: Bright | Medium | Dark ", np.hstack((bri, med, drk)))
#     cv2.imshow("LAB: Bright | Medium | Dark ", np.hstack((lab_bri, lab_med, lab_drk)))

    ## === Subtract the luminosity channel
    # medium to bright
    m2b = lab_bri[:,:,0] - lab_med[:,:,0]
    # medium to dark
    m2d = lab_med[:,:,0] - lab_drk[:,:,0]
    
#     var = m2b
#     print var.shape, var.min(), var.max()
#     var = m2d
#     print var.shape, var.min(), var.max()
    imnames = imt.get_imlist(root,ext ='.png')
    for src in imnames:
        if (view in src) and ("medium" in src) and ("rgb" in src) and not ("clear" in src):
            # Generate bright scene unsing medium
            med = cv2.imread(src)
            lab_med = cv2.cvtColor(med, cv2.cv.CV_RGB2Lab)
            M2B = lab_med.copy()
            M2B[:,:,0]= np.uint8(M2B[:,:,0] + np.median(m2b, axis =1))
#             M2B[:,:,0]= np.uint8(M2B[:,:,0] + np.max(m2b, axis =0))
            
#             M2B[:,:,0] = M2B[:,:,0] + m2b
#             M2B[:,:,0] = 100*(M2B[:,:,0]/np.max(M2B[:,:,0]))
#             M2B = np.uint8(255*M2B.astype(float)/np.max(M2B))
#             M2B[M2B > 255] = 255
            M2B = cv2.cvtColor(M2B, cv2.cv.CV_Lab2RGB)
#             M2B = np.uint8((M2B.astype(float)/M2B.max())* 255)
            dst_bri = src.replace("medium", "bright")
            # Generate dark scene unsing medium
#             M2D=np.zeros(lab_med.copy().shape, dtype = lab_med.dtype)
#             M2D[:,:,0] = M2D[:,:,0] - m2d.max()
#             M2D[M2D <0] =0
#             M2D =cv2.cvtColor(M2D, cv2.cv.CV_Lab2RGB)
            M2D=np.zeros(lab_med.copy().shape, dtype = lab_med.dtype)
            
            dst_drk = src.replace("medium", "dark")
            
#             med = cv2.cvtColor(med, cv2.COLOR_RGB2GRAY)
            print "NAMES medium:{} \t brigh:{} \t dark:{}".format(src, dst_bri, dst_drk)
            cv2.imshow("LAB: Medium | New Bright | New Dark ", np.hstack((med, M2B, M2D)))
            cv2.waitKey(15)
#             cv2.imwrite(dst_bri, M2B)
#             cv2.imwrite(dst_drk, M2D)
#generate_synthetic_dark_bright_rgb_scenes            

def generate_synthetic_dark_bright_depth_pressure_scenes():
    """
    Copy with new label (Illumination invariant) scenes
    """
    root = "../data/framed"
    modality = "depth" #"pressure"
    imnames = imt.get_imlist(root,ext ='.png')
    for src in imnames:
        if("medium" in src) and ("binary" in src) and not ("clear" in src):

            dst_bri = src.replace("medium", "bright")
            
            dst_drk = src.replace("medium", "dark")
            
#             med = cv2.cvtColor(med, cv2.COLOR_RGB2GRAY)
            print "NAMES medium:{} \t brigh:{} \t dark:{}".format(src, dst_bri, dst_drk)            
            shutil.copy(src,  dst_bri)
            shutil.copy(src,  dst_drk)
            
## ==========================================================
#manicure_data()
#generate_pose_folders()
#copy2organized_folders()


#replicate_and_rename_binary_images()
#replicate_and_rename_pressure_images()
#replicate_and_rename_rgbd_images()


# copy_all_images_to_single_folder()

# resize_rotate_and_frame()
#binarize_images_under_folder()

#inverse_pressure_images()


## Steps to prepare pressure images:
#b = imt.shrinkIm(im, s)


# correct_rotation_of_right_view()

# adjust_dataset_names()
generate_synthetic_dark_bright_rgb_scenes()
# generate_synthetic_dark_bright_depth_pressure_scenes()

print "CODE EXECUTION COMPLETE!!"
