import numpy as np
from monk import BBox
import tensorflow as tf
from monk import Dataset
import json
import PIL
from monk.utils.s3.s3path import S3Path
from monk import Dataset, PredictorPipe, cfg_static_default, Image

class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self,json_paths, batch_size=10, dim=(128,128), n_channels=3,shuffle=True,damaged=False):
        
        
        
        self.shuffle = shuffle 
        self.dim = dim 
        self.batch_size = batch_size  
        self.n_channels = n_channels
        self.damaged=damaged
        
        jsons_data=[]

        cfg =  cfg_static_default.clone()
        self.pipe = PredictorPipe.from_config(cfg)
        
        
        for json_path in json_paths:
            with open(json_path) as f:
                json_data = json.load(f)
            jsons_data.append(json_data)
            
        self.filter_json(jsons_data,damaged)

        
        
    def filter_json(self,jsons_data,damaged):
        
        
        filtered_json =[]

        for json_data in jsons_data :
        
            for i in range(0,len(json_data)):
                
                
                if json_data[i]["repair_action"]=='not_damaged' and damaged==False :
                    filtered_json.append(json_data[i])
                elif json_data[i]["repair_action"]!='not_damaged' and json_data[i]["repair_action"]!=None and damaged==True and (json_data[i]["label"]=='scratch' or json_data[i]["label"]=='dent' ) :
                    filtered_json.append(json_data[i])
               
        
        self.filtered_json=filtered_json
        self.list_IDs = np.arange(len(filtered_json)) 
        self.indexes = np.arange(len(filtered_json))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        

    
    def load_image(self,id):
        
        data = self.filtered_json[id]
        
        
        if('s3:/monk-client-images/' in data["path"]):
            bucket = "monk-client-images"
            key = data["path"].replace("s3:/monk-client-images/","")
            s3 = S3Path(bucket,key)
            im = PIL.Image.open(s3.download())
        else:
            im = PIL.Image.open(data["path"])
        
        
        part = data["part"]
        box = data["part_bbox"]
        im = np.array(im).astype('uint8')
        image = Image(rgb=im)
        res = self.pipe.infer_parts(image)
        #mask =res.get_masks([part])[0]
        #mask = mask.dilate(50)

        mask =res.get_masks(["background"])[0]
        mask_part =res.get_masks([part])[0]

        if not mask_part.is_empty():
            bb_xyxy = mask_part.to_bbox().xyxy
    
        else:
            bb_xyxy=[0,1,0,1]


        mask3d = np.moveaxis(np.array([mask_part.arr,mask_part.arr,mask_part.arr]),0,2)
        img_masked = res.image.arr*(mask3d)
        img_masked = np.array(img_masked).astype("uint8")
        ## get part
        img_masked = PIL.Image.fromarray(img_masked)
        img_crop = img_masked.crop(bb_xyxy)
        img_crop= img_crop.resize(self.dim)
        
        #img_crop = img_masked[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
        #img_crop = img_masked
       
        #print(bbox)

        #img_crop = np.resize(img_crop,(self.dim[0],self.dim[1],3))

        
        
        return(  ((((np.array(img_crop)/255)*2)-1)).astype(np.float32))
        #return(np.array(img_crop).astype(np.float32))
        
    def __len__(self):
       
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
    
        
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs_temp)

        return tf.convert_to_tensor(X,dtype=tf.float32)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels),dtype=np.float32)
        #y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = self.load_image(self.indexes[ID])

            # Store class
            #y[i] = self.labels[ID]

        #return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        return tf.convert_to_tensor(X)

def get_generator(json_paths,batch_size,size,damaged=False):
    
    generator = DataGenerator(json_paths,batch_size=batch_size,dim=(size,size),damaged=damaged)

    return(generator)