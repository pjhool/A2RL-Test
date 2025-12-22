# -*- coding: utf-8 -*-
import numpy as np
import skimage.transform as transform

import skimage.io as io

def command2action(command_ids, ratios, terminals):

    #print ( ' command2action command_ids = ' , command_ids )
    #print(' command2action command_ids[0] = ', command_ids[0])
    batch_size = len(command_ids)
    #print (' command2action batch_size =' , batch_size)
    for i in range(batch_size):
        if terminals[i] == 1:
            continue
        if command_ids[i] == 0:
            ratios[i, 0] += 1
            ratios[i, 1] += 1
            ratios[i, 2] -= 1
            ratios[i, 3] -= 1
        elif command_ids[i] == 1:
            ratios[i, 2] -= 1
            ratios[i, 3] -= 1
        elif command_ids[i] == 2:
            ratios[i, 0] += 1
            ratios[i, 3] -= 1
        elif command_ids[i] == 3:
            ratios[i, 1] += 1
            ratios[i, 2] -= 1
        elif command_ids[i] == 4:
            ratios[i, 0] += 1
            ratios[i, 1] += 1
        elif command_ids[i] == 5:
            ratios[i, 0] += 1
            ratios[i, 2] += 1
        elif command_ids[i] == 6:
            ratios[i, 0] -= 1
            ratios[i, 2] -= 1
        elif command_ids[i] == 7:
            ratios[i, 1] -= 1
            ratios[i, 3] -= 1
        elif command_ids[i] == 8:
            ratios[i, 1] += 1
            ratios[i, 3] += 1
        elif command_ids[i] == 9:
            ratios[i, 1] += 1
            ratios[i, 3] -= 1
        elif command_ids[i] == 10:
            ratios[i, 0] += 1
            ratios[i, 2] -= 1
        elif command_ids[i] == 11:
            ratios[i, 1] -= 1
            ratios[i, 3] += 1
        elif command_ids[i] == 12:
            ratios[i, 0] -= 1
            ratios[i, 2] += 1
        elif command_ids[i] == 13:
            terminals[i] = 1
        else:
            raise NameError('undefined command type !!!')
        #print ( ' before action ratios =' , ratios)
        ratios = np.maximum(ratios, 0)
        ratios = np.minimum(ratios, 20)
        #print(' after  action ratios =', ratios)
        if ratios[i, 2] - ratios[i, 0] <= 4 or ratios[i, 3] - ratios[i, 1] <= 4:
            terminals[i] = 1

    return ratios, terminals

def generate_bbox(input_np, ratios):
    assert len(input_np) == len(ratios)

    bbox = []
    for im, ratio in zip(input_np, ratios):
        height, width = im.shape[:2]
        xmin = int(float(ratio[0]) / 20 * width)
        ymin = int(float(ratio[1]) / 20 * height)
        xmax = int(float(ratio[2]) / 20 * width)
        ymax = int(float(ratio[3]) / 20 * height)
        
        # Ensure coordinates are within image bounds
        xmin = max(0, min(xmin, width - 1))
        ymin = max(0, min(ymin, height - 1))
        xmax = max(0, min(xmax, width))
        ymax = max(0, min(ymax, height))
        
        # Ensure minimum width and height (at least 5 pixels)
        if xmax - xmin < 5:
            xmax = min(xmin + 5, width)
            xmin = max(0, xmax - 5)
        
        if ymax - ymin < 5:
            ymax = min(ymin + 5, height)
            ymin = max(0, ymax - 5)
        
        bbox.append((xmin, ymin, xmax, ymax))
    #print( ' generate bbox ==', bbox)
    return bbox

my_static = None
def crop_input(input_np, bbox):
        assert len(input_np) == len(bbox)
        global my_static
        if my_static is None:
            my_static = 0
        else:
            my_static = my_static + 1
        #fileName = 'cropBox/' + my_static + '.jpg'
        #print(fileName)
        #io.imsave( fileName, im[ymin:ymax, xmin:xmax])  for im, (xmin, ymin, xmax, ymax) in zip(input_np, bbox)
        
        result = []
        for im, (xmin, ymin, xmax, ymax) in zip(input_np, bbox):
            # Validate bounding box dimensions
            crop_width = xmax - xmin
            crop_height = ymax - ymin
            
            if crop_width <= 0 or crop_height <= 0:
                print("WARNING: Invalid crop dimensions ({}x{}), using full image".format(
                    crop_width, crop_height))
                # Use full image as fallback
                cropped = im
            else:
                # Crop the image
                cropped = im[ymin:ymax, xmin:xmax]
                
                # Additional check after cropping
                if cropped.shape[0] == 0 or cropped.shape[1] == 0:
                    print("WARNING: Empty crop result, using full image")
                    cropped = im
            
            # Resize to target size
            resized = transform.resize(cropped, (227, 227), mode='constant')
            result.append(resized)
    
        return np.asarray(result, dtype=np.float32)