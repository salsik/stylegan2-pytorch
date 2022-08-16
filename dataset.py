from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset

import pyarrow
import lz4framed

import os
from typing import Any



class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)


   

        # this is to print first 100 key val 
        self.keys = []
        with self.env.begin() as txn:
            ii = txn.cursor()
            print("oqioeiqoeioqioeq",ii)
            cc=0
            for key,val in txn.cursor():
                self.keys.append(key)
                #print(key)
                cc+=1
                if (cc>100000):
                    break
           
            #myList = [ key for key, _ in txn.cursor() ]
            #print(myList)




        with self.env.begin(write=False) as txn:

            ## this way we get all number of entries
            length = txn.stat()['entries']

            # this way we are sure that there is an entrie named "length" which contains the total number of entries,
            # that's why number of entries is 101 and key length has 100 as value
            #self.length =  int(txn.get('length'.encode('utf-8')).decode('utf-8'))

            # we will use this from now and on, which is common in all lmdb files
            # if we have extra enty we put length -1 


            self.length = txn.stat()['entries'] - 1
            #self.keys = pyarrow.deserialize(lz4framed.decompress(txn.get(b'__keys__')))
            
            self.length =len(self.keys) -1
            print(f"Total records: {len(self.keys), self.length}")

        

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            #key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            #img_bytes = txn.get(key)
    
    
            img_bytes= txn.get(self.keys[index])

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img


    def __getitem22__(self, index):
        lmdb_value = None
        with self.lmdb_connection.begin(write=False) as txn:
            lmdb_value = txn.get(self.keys[index])
        assert lmdb_value is not None, f"Read empty record for key: {self.keys[index]}"

        img_name, img_arr, img_shape = LMDBDataset.decompress_and_deserialize(lmdb_value=lmdb_value)
        image = np.frombuffer(img_arr, dtype=np.uint8).reshape(img_shape)
        if image.size == 0:
            raise InvalidFileException("Invalid file found, skipping")
        return image



class LMDBDataset(Dataset):
    def __init__(self, lmdb_store_path, transform=None):
        super().__init__()
        assert os.path.isfile(lmdb_store_path), f"LMDB store '{lmdb_store_path} does not exist"
        assert not os.path.isdir(lmdb_store_path), f"LMDB store name should a file, found directory: {lmdb_store_path}"

        self.lmdb_store_path = lmdb_store_path
        self.lmdb_connection = lmdb.open(lmdb_store_path,
                                         subdir=False, readonly=True, lock=False, readahead=False, meminit=False)

        with self.lmdb_connection.begin(write=False) as lmdb_txn:
            self.length = lmdb_txn.stat()['entries'] - 1
            self.keys = pyarrow.deserialize(lz4framed.decompress(lmdb_txn.get(b'__keys__')))
            print(f"Total records: {len(self.keys), self.length}")
        self.transform = transform

    def __getitem__(self, index):
        lmdb_value = None
        with self.lmdb_connection.begin(write=False) as txn:
            lmdb_value = txn.get(self.keys[index])
        assert lmdb_value is not None, f"Read empty record for key: {self.keys[index]}"

        img_name, img_arr, img_shape = LMDBDataset.decompress_and_deserialize(lmdb_value=lmdb_value)
        image = np.frombuffer(img_arr, dtype=np.uint8).reshape(img_shape)
        if image.size == 0:
            raise InvalidFileException("Invalid file found, skipping")
        return image

    @staticmethod
    def decompress_and_deserialize(lmdb_value: Any):
        return pyarrow.deserialize(lz4framed.decompress(lmdb_value))

    def __len__(self):
        return self.length