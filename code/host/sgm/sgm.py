import cv2
import numpy as np
from numba import jit
import time
from scipy.ndimage import median_filter
import os
# 计算 x 和 y 方向上的 Sobel 梯度

def compute_gradient(image):
    grad_x = cv2.Sobel(image, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_16S, 0, 1, ksize=3)
    
    # 计算梯度幅值
    gradient = cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5,
                               cv2.convertScaleAbs(grad_y), 0.5, 0)
    return grad_x, grad_y, gradient*2

@jit(nopython=True)
def census_transform(image):
    height, width = image.shape
    census = np.zeros((height, width), dtype=np.uint8)
    
    # 对每个像素进行 Census 变换
    for row in range(1, height - 1):
        for col in range(1, width - 1):
            center = image[row, col]
            bit_string = 0
            for r in range(-1, 2):
                for c in range(-1, 2):
                    if r == 0 and c == 0:
                        continue
                    bit_string <<= 1
                    if image[row + r, col + c] < center:
                        bit_string |= 1
            census[row, col] = bit_string

    return census

@jit(nopython=True)
def hamming_distance(val1, val2):
    xor =val1^val2
    distance=0
    while xor>0:
        distance+=xor&1
        xor>>=1
    return distance

@jit(nopython=True)
def exponential_cost(value):
    
    # 模拟 exponentialCost 函数
    value=abs(value)
    if value >= 32:
        return 0
    elif 0 <= value < 0.125:
        return -0.875 * value + 1
    elif 0.125 <= value < 0.25:
        return -0.8125 * value + 0.9375
    elif 0.25 <= value < 0.5:
        return -0.6875 * value + 0.9375
    elif 0.5 <= value < 1:
        return -0.4375 * value + 0.8125
    elif 1 <= value < 1.5:
        return -0.25 * value + 0.625
    elif 1.5 <= value < 2:
        return -0.125 * value + 0.4375
    elif 2 <= value < 3:
        return -0.0625 * value + 0.25
    elif 3 <= value < 4:
        return -0.03125 * value + 0.140625
    elif 4 <= value < 5:
        return 0.0625
    elif 5 <= value < 6:
        return 0.015625
    return 0

@jit(nopython=True)
def hamming_trans(xor):
    distance=0
    while xor>0:
        distance+=xor&1
        xor>>=1
    return distance

@jit(nopython=True)
def hamming_trans_all(array):
    cases=np.array([1,0.9375,0.875,0.828125,0.765625,0.71875,0.671875,0.640625,0.59375])
    rows,cols=array.shape
    results=np.zeros_like(array,dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            index=hamming_trans(array[row,col])
            results[row,col]=cases[index]
    return results 


@jit(nopython=True)
def calculate_pixel_cost(leftimage_grad, rightimage_grad,
                         left_grad_x, left_grad_y, right_grad_x, right_grad_y,
                         disparity_range, Grad, Ctg, Window_Size):
    # 处理左图和右图，得到 Census 图像
    left_census = census_transform(leftimage_grad)
    right_census = census_transform(rightimage_grad)

    img_height, img_width = leftimage_grad.shape
    cost_matrix = np.full((img_height, img_width, disparity_range//4),Window_Size*Window_Size,dtype=np.float32)
    index_matrix = np.zeros((img_height, img_width, disparity_range//4),dtype=np.float32)
    cases=np.array([1,0.9375,0.875,0.828125,0.765625,0.71875,0.671875,0.640625,0.59375])
    bit = 3

    for i in range(img_height):
        for j in range(3,img_width):
            L = left_census[i, j]
            if j < 256:
                valid_range=(j+1)//4
            else:
                valid_range=disparity_range//4

            for k in range(valid_range):
                temp = 4 * k
                temp1, temp2, temp3 = temp + 1, temp + 2, temp + 3
                y1, y2, y3, y4 = j - temp, j - temp1, j - temp2, j - temp3

                
                R1 = right_census[i, y1]
                R2 = right_census[i, y2]
                R3 = right_census[i, y3]
                R4 = right_census[i, y4]

                hamming = [hamming_distance(L,R) for R in [R1, R2, R3, R4]]
                    
                    # 计算梯度差异
                cost_grad = [
                        abs(int(left_grad_x[i, j]) - int(right_grad_x[i, y])) +
                        abs(int(left_grad_y[i, j]) - int(right_grad_y[i, y]))
                        for y in [y1, y2, y3, y4]
                ]
                    
                    # 计算代价 hamming 权重
                    
                cost_hamming = [cases[h] for h in hamming]
                    
                    # 计算 cost_exponent
                cost_exponent = [-(cost / float(Grad)) for cost in cost_grad]
                    
                    # 插值计算代价
                cost_1 = [exponential_cost(exp) for exp in cost_exponent]
                    
                    
                cost_temp = [2 - c1 - ch for c1, ch in zip(cost_1, cost_hamming)]
                cost = [float(int(ct * (2 ** bit))) / (2 ** bit) for ct in cost_temp]

                    # 找到最小代价并保存
                min_cost = min(cost)
                min_index = cost.index(min_cost)

                index_matrix[i][j][k] = min_index
                cost_matrix[i][j][k] = min_cost
    return index_matrix, cost_matrix

@jit(nopython=True)
def calculate_pixel_cost_R(leftimage_grad, rightimage_grad, 
                           left_grad_x, left_grad_y, right_grad_x, right_grad_y,
                           disparity_range, Grad, Ctg, Window_Size):
    # 处理左图和右图，得到 Census 图像
    left_census = census_transform(leftimage_grad)
    right_census = census_transform(rightimage_grad)

    img_height, img_width = rightimage_grad.shape
    cost_matrix = np.full((img_height, img_width, disparity_range//4),Window_Size*Window_Size,dtype=np.float32)
    index_matrix = np.zeros((img_height, img_width, disparity_range//4),dtype=np.float32)
    cases=np.array([1,0.9375,0.875,0.828125,0.765625,0.71875,0.671875,0.640625,0.59375])
    bit = 3

    for i in range(img_height):
        for j in range(img_width-3):
            R = right_census[i, j]
            if j > img_width-256:
                valid_range=(img_width-j)//4
            else:
                valid_range=disparity_range // 4

            for k in range(valid_range):
                temp = 4 * k
                temp1, temp2, temp3 = temp + 1, temp + 2, temp + 3
                y1, y2, y3, y4 = j + temp, j + temp1, j + temp2, j + temp3

                L1 = left_census[i, y1]
                L2 = left_census[i, y2]
                L3 = left_census[i, y3]
                L4 = left_census[i, y4]

                hamming = [hamming_distance(R,L) for L in [L1, L2, L3, L4]]
                    # 计算梯度差异
                cost_grad = [
                        abs(int(right_grad_x[i, j]) - int(left_grad_x[i, y])) +
                        abs(int(right_grad_y[i, j]) - int(left_grad_y[i, y]))
                        for y in [y1, y2, y3, y4]
                    ]
                    
                    # 计算代价 hamming 权重
                cost_hamming = [cases[h] for h in hamming]
                    
                    # 计算 cost_exponent
                cost_exponent = [-(cost / float(Grad)) for cost in cost_grad]
                    
                    # 插值计算代价
                cost_1 = [exponential_cost(exp) for exp in cost_exponent]
                    
                cost_temp = [2 - c1 - ch for c1, ch in zip(cost_1, cost_hamming)]
                cost = [float(int(ct * (2 ** bit))) / (2 ** bit) for ct in cost_temp]

                    # 找到最小代价并保存
                min_cost = min(cost)
                min_index = cost.index(min_cost)

                index_matrix[i][j][k] = min_index
                cost_matrix[i][j][k] = min_cost

        # 打印进度
    #     progress = i * 100.0 / (img_height - 1)
    #     print(f"\rcalculatePixelCost_R Progress: {progress:.2f}%", end='')
    # print("\n")
    return index_matrix, cost_matrix

@jit(nopython=True)     
def grad_cost(grad,Grad):
    rows,cols=grad.shape
    results=np.zeros_like(grad,dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            cost_exponent=-(grad[row,col]/float(Grad))
            results[row,col]=exponential_cost(cost_exponent)
    return results

    
@jit(nopython=True)
def calculate_pixel_cost_all(leftimage_grad, rightimage_grad,
                         left_grad_x, left_grad_y, right_grad_x, right_grad_y,
                         disparity_range, Grad, Ctg, Window_Size):
    # 处理左图和右图，得到 Census 图像
    left_census = census_transform(leftimage_grad)
    right_census = census_transform(rightimage_grad)

    img_height, img_width = leftimage_grad.shape
    cost_matrix_left = np.full((img_height, img_width, disparity_range//4),Window_Size*Window_Size,dtype=np.float32)
    index_matrix_left = np.zeros((img_height, img_width, disparity_range//4),dtype=np.float32)
    cost_matrix_right = np.full((img_height, img_width, disparity_range//4),Window_Size*Window_Size,dtype=np.float32)
    index_matrix_right = np.zeros((img_height, img_width, disparity_range//4),dtype=np.float32)
    
    bit = 3
    num_bit=(2 ** bit)
    
    
    print("calculatePixelCost for left......")
    final_cost_left=np.zeros((img_height, img_width, disparity_range),dtype=np.float32)
    hamming_cost_left=hamming_trans_all(left_census^right_census)
    gradient_cost_left=grad_cost(np.abs(left_grad_x-right_grad_x)+np.abs(left_grad_y-right_grad_y),Grad)
    cost_temp=(2-hamming_cost_left-gradient_cost_left)*num_bit
    final_cost_left[:,:,0]=(cost_temp.astype(np.uint16)).astype(np.float32)/ num_bit

    for i in range(1,disparity_range):
        hamming_cost_left = hamming_trans_all(left_census[:,i:] ^ right_census[:,:-i])
        gradient_cost_left=grad_cost(np.abs(left_grad_x[:,i:]-right_grad_x[:,:-i])+np.abs(left_grad_y[:,i:]-right_grad_y[:,:-i]),Grad)
        cost_temp=(2-hamming_cost_left-gradient_cost_left)* num_bit
        final_cost_left[:,i:,i]=(cost_temp.astype(np.uint16)).astype(np.float32)/ num_bit
    
    for i in range(img_height):
        for j in range(3,img_width):
            if j < 256:
                valid_range=(j+1)//4
            else:
                valid_range=disparity_range//4
                
            for k in range(valid_range):
                temp = 4 * k
                cost=[final_cost_left[i,j,temp],
                      final_cost_left[i,j,temp+1],
                      final_cost_left[i,j,temp+2],
                      final_cost_left[i,j,temp+3]]

                # 找到最小代价并保存
                min_cost = min(cost)
                min_index = cost.index(min_cost)

                index_matrix_left[i,j,k] = min_index
                cost_matrix_left[i,j,k] = min_cost
                
    print("calculatePixelCost for right......")
    for i in range(img_height):
        for j in range(img_width-3):
            if j > img_width-256:
                valid_range=(img_width-j)//4
            else:
                valid_range=disparity_range // 4

            for k in range(valid_range):
                temp = 4 * k
                col = j + temp
                
                cost = [final_cost_left[i,col,temp],
                        final_cost_left[i,col+1,temp+1],
                        final_cost_left[i,col+2,temp+2],
                        final_cost_left[i,col+3,temp+3]]

                # 找到最小代价并保存
                min_cost = min(cost)
                min_index = cost.index(min_cost)

                index_matrix_right[i,j,k] = min_index
                cost_matrix_right[i,j,k] = min_cost
    return index_matrix_left, cost_matrix_left,index_matrix_right, cost_matrix_right



@jit(nopython=True)
def aggregate_cost(row, col, d, rows, cols, disparity_range, cost, disparity_i,min_value_i,LARGE_PENALTY,  SMALL_PENALTY):


    prev_plus=prev_minus=prev=disparity_i[d]
    if d<disparity_range//4-1:
        prev_plus=disparity_i[d+1] + SMALL_PENALTY
    if d>0:
        prev_minus=disparity_i[d-1] + SMALL_PENALTY
    min_prev_other=min_value_i + LARGE_PENALTY
    
    aggregated_cost = cost
    aggregated_cost += min(
        min(prev_plus, prev_minus),
        min(prev, min_prev_other)
    )
    aggregated_cost -= min_value_i

    return aggregated_cost


@jit(nopython=True)
def aggregate_costs_0(rows, cols, disparity_range, C, LARGE_PENALTY,  SMALL_PENALTY):
    result=np.zeros((rows,cols,disparity_range//4),dtype=np.float32)
    disparity_i=np.zeros(disparity_range//4,dtype=np.float32)
    disparity_o=np.zeros(disparity_range//4,dtype=np.float32)
    min_value_o=0
    result[:,0]=C[:,0]
    # last_progress_printed = 0
    print("Aggregation0 start...")
    for row in range(rows):
        disparity_i[:]=C[row,0]
        min_value_i=disparity_i.min()
        for col in range(1,cols):
            for d in range(disparity_range // 4):
                result_tmp= aggregate_cost(row, col, d, rows, cols, disparity_range, C[row,col,d], disparity_i
                                                ,min_value_i, LARGE_PENALTY,  SMALL_PENALTY)
                result[row, col, d]=result_tmp
                disparity_o[d]=result_tmp
                if d==0 or (result_tmp < min_value_o):
                    min_value_o=result_tmp
            min_value_i=min_value_o
            disparity_i[:]=disparity_o
    #     last_progress_printed = print_progress(row, rows - 1, last_progress_printed)
    # print("")
    return result

@jit(nopython=True)
def find_min(array):
    size,_=array.shape
    minv=np.zeros(size)
    i=0
    for v in array:
        minv[i]=v.min()
        i+=1
    return minv

@jit(nopython=True)
def aggregate_costs_135(rows, cols, disparity_range, C, LARGE_PENALTY,  SMALL_PENALTY):
    result=np.zeros((rows,cols,disparity_range//4),dtype=np.float32)
    disparity_i=np.zeros((cols,disparity_range//4),dtype=np.float32)
    disparity_o=np.zeros((cols,disparity_range//4),dtype=np.float32)
    min_value_i=np.zeros(cols,dtype=np.float32)
    min_value_o=np.zeros(cols,dtype=np.float32)
    # last_progress_printed = 0
    print("Aggregation135 start...")
    result[0] = C[0]
    result[:,cols-1]=C[:,cols-1]
    
    disparity_i[:]=C[0]
    min_value_i=find_min(disparity_i)
    
    for row in range(1,rows):
        for col in range(cols-1):
            for d in range(disparity_range // 4):
                result_tmp= aggregate_cost(row, col, d, rows, cols, disparity_range, C[row,col,d], disparity_i[col+1]
                                                ,min_value_i[col+1], LARGE_PENALTY,  SMALL_PENALTY)
                result[row, col, d]=result_tmp
                disparity_o[col,d]=result_tmp
                if d==0 or (result_tmp < min_value_o[col]):
                    min_value_o[col]=result_tmp
        min_value_i[:]=min_value_o
        disparity_i[:]=disparity_o
        
        disparity_i[cols-1]=C[row,cols-1]
        min_value_i[cols-1]=disparity_i[cols-1].min()

    #     last_progress_printed = print_progress(row, rows - 1, last_progress_printed)
    # print("")
    return result    

@jit(nopython=True)
def compute_disparity(S, rows, cols, disparity_range, B):
    disparity_map = np.zeros((rows, cols), dtype=np.float32)  # 初始化视差图
    for row in range(rows):
        for col in range(cols):
            min_cost = 99 # 替代 MAX_SHORT
            disparity1 = 0
            th = disparity_range // 4 - 1
            min_disp=th
            
            for d in range(disparity_range // 4 - 1, -1, -1):
                if (min_cost > S[row][col][d]):
                    min_cost = S[row][col][d]
                    min_disp=d
                    
            x = 0
            interp_function = 0

            if 0 < min_disp < th:
                left_dif = S[row][col][min_disp - 1] - S[row][col][min_disp]
                right_dif = S[row][col][min_disp + 1] - S[row][col][min_disp]
                        
                if left_dif == 0:
                    x = 0
                else:
                    x = right_dif / left_dif
                        
                interp_function = (x * x + x) / 4
                if 0 < interp_function < 1:
                    if left_dif <= right_dif:
                        disparity1 = 4 * min_disp + B[row][col][min_disp] - 0.5 + interp_function
                    else:
                        disparity1 = 4 * min_disp + B[row][col][min_disp] + 0.5 - interp_function
                else:
                    disparity1 = 4 * min_disp + B[row][col][min_disp]
            else:
                disparity1 = 4 * min_disp + B[row][col][min_disp]
            
            disparity_map[row, col] = disparity1 
    
    print("Disparity finish")
    return disparity_map

def compute_disparity2(S, rows, cols, disparity_range, B):
    disparity_map = np.zeros((rows, cols), dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            disp=S[row,col]
            min_disp=np.argmin(disp)
            disparity_map[row,col]=4*min_disp+B[row,col,min_disp]
    return disparity_map

def compute_disparity3(S, rows, cols, disparity_range):
    disparity_map = np.zeros((rows, cols), dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            disp=S[row,col]
            min_disp=np.argmin(disp)
            disparity_map[row,col]=min_disp
    return disparity_map

@jit(nopython=True)
def left_right_check_window(disparity_map, disparity_map_r, window_size2):
    row, col = disparity_map.shape
    occlusion=np.zeros((row, col),dtype=np.bool_)
    mismatches=np.zeros((row, col),dtype=np.bool_)
    disparity_map_hole=np.zeros((row, col),dtype=np.float32)
    for j in range(row):
        for i in range(col):
            dL = value1 = disparity_map[j, i]
            # window_start1 = max(i - window_size1 // 2, 0)
            # window_end1 = min(i + window_size1 // 2, col - 1)
            window_start2 = int(i - dL) - window_size2 // 2
            window_end2 = int(i - dL) + window_size2 // 2
            
            if window_start2 >= 0 and window_end2 < col:
                total_sum = 0
                value = 0

                # Compare elements in the window and calculate the total sum
                # for k in range(window_start1, window_end1 + 1):
                for l in range(window_start2, window_end2 + 1):
                    value2 = disparity_map_r[j, l]
                    total_sum += abs(value1 - value2)
                    value += value1 - value2

                # Update output matrices based on whether the total sum exceeds the threshold
                if total_sum / ( window_size2) > 8:
                    if (value / (window_size2)) > 0:
                        occlusion[j, i] = True
                    else:
                        mismatches[j, i] = True
                    disparity_map_hole[j, i] = 0
                else:
                    occlusion[j, i] = False
                    mismatches[j, i] = False
                    disparity_map_hole[j, i] = disparity_map[j, i]   # *2 for new filling
            else:
                mismatches[j, i] = True
    print("disparity left_right_check finish")
    return occlusion,mismatches,disparity_map_hole
@jit(nopython=True)
def left_right_check_window3(disparity_map, disparity_map_r, window_size2):
    row, col = disparity_map.shape
    occlusion=np.zeros((row, col),dtype=np.uint8)
    mismatches=np.zeros((row, col),dtype=np.uint8)
    disparity_map_hole=np.zeros((row, col),dtype=np.float32)
    for j in range(row):
        for i in range(col):
            #dL = value1 = disparity_map[j, i]
            dL = value1 = disparity_map[j, i]
            window_start1 = max(int(j - window_size2 // 2), 0)
            window_end1 = min(int(j + window_size2 // 2), row - 1)
            # window_start2 = int(i - dL) - window_size2 // 2
            # window_end2 = int(i - dL) + window_size2 // 2
            index_r=int(i-dL)
            
            # if window_start2 >= 0 and window_end2 < col:
            #     total_sum = 0
            #     value = 0
            total_sum=0
            value=0
            
            if index_r >= 0:
                # Compare elements in the window and calculate the total sum
                for k in range(window_start1, window_end1 + 1):
                    # for l in range(window_start2, window_end2 + 1):
                        value2 = disparity_map_r[k, index_r]
                        total_sum += abs(value1 - value2)
                        #value += value3-value1
                        value += value1-value2
    
                    # Update output matrices based on whether the total sum exceeds the threshold
                    # if total_sum / ( window_size2) > 8:
                if total_sum / ( window_size2) > 8:
                        if (value>0) :
                        # if (xl-dL > 0) and (value1!=0) and (int(index_r+x)<col-128):
                            occlusion[j, i] = 1
                        else:
                            mismatches[j, i] = 1
                        disparity_map_hole[j, i] = 0
                else:
                        occlusion[j, i] = 0
                        mismatches[j, i] = 0
                        disparity_map_hole[j, i] = disparity_map[j, i]   # *2 for new filling
            else:
                mismatches[j, i] = 1
    print("disparity left_right_check finish")
    return occlusion,mismatches,disparity_map_hole
@jit(nopython=True)
def filling2(disparity_map, disparity_map_r, occlusion, mismatches):
    row, col = disparity_map.shape
    disparity_final=np.zeros((row,col),dtype=np.float32)
    directions = [
        (-1, 0),  # left
        (-1, -1),  # top-left diagonal
        (0, -1),  # top
        (1, -1),  # top-right diagonal
        (1, 0)  # right
    ]

    for j in range(row):
        for i in range(col):
            if occlusion[j, i] :
                values = np.zeros(5,dtype=np.float32)
                index=0
                for dx, dy in directions:
                    y, x = j+dy, i+dx
                    while 0 <= y < row and 0 <= x < col:
                        if not mismatches[y, x] and not occlusion[y, x]:
                            values[index]=disparity_map[y, x]
                            index += 1
                            break
                        y += dy
                        x += dx

                if index == 1 :
                    disparity_final[j, i] =values[0]
                else:
                    disparity_final[j, i] = sorted(values)[1]  # Second smallest value

            elif mismatches[j, i]:
                values = []
                for dx, dy in directions:
                    y, x = j+dy, i+dx
                    while 0 <= y < row and 0 <= x < col:
                        if not mismatches[y, x]  and not occlusion[y, x] :
                            values.append(disparity_map[y,x])
                            break
                        y += dy
                        x += dx
                        
                length=len(values)   
                
                if length==1:
                    disparity_final[j, i] = values[0]
                elif length==2:
                    disparity_final[j, i] = max(values)
                elif length==3:
                    disparity_final[j, i] = sorted(values)[1]
                else:
                    disparity_final[j, i] = sorted(values)[2]  # Median value

            else:
                disparity_final[j, i] = disparity_map[j, i]
    print("filling finish")
    return disparity_final

@jit(nopython=True)
def filling_window2(disparity_map, disparity_map_r, occlusion, mismatches,window_size): 
    # Ensure window_size is odd
    window_size = window_size | 1  # Make window_size odd if it's even
    
    row, col = disparity_map.shape
    half_window = window_size // 2
    disparity_final=np.zeros((row,col),dtype=np.float32)

    for j in range(half_window, row - half_window):
        for i in range(half_window, col - half_window):
            # Define a window for the current point
            valid_disparities = []
            for wj in range(-half_window, half_window + 1):
                for wi in range(-half_window, half_window + 1):
                    if not mismatches[j + wj, i + wi]  and not occlusion[j + wj, i + wi] :
                        valid_disparities.append(disparity_map[j + wj, i + wi])

            # Calculate median
            median = 0.0
            if valid_disparities:
                valid_disparities.sort()
                n = len(valid_disparities) // 2
                if len(valid_disparities) % 2 == 0:
                    median = 0.5 * (valid_disparities[n] + valid_disparities[n - 1])
                else:
                    median = valid_disparities[n]

            # Set the final disparity value based on conditions
            if occlusion[j, i]:
                disparity_final[j, i] = median
                occlusion[j, i] = False
            elif mismatches[j, i]:
                disparity_final[j, i] = median
                mismatches[j, i] = False
            else:
                disparity_final[j, i] = disparity_map[j, i]
    

    # Copy border areas directly from disparity_map to disparity_final
    disparity_final[:half_window, :] = disparity_map[:half_window, :]
    disparity_final[-half_window:, :] = disparity_map[-half_window:, :]
    disparity_final[:, :half_window] = disparity_map[:, :half_window]
    disparity_final[:, -half_window:] = disparity_map[:, -half_window:]

    print("filling finish")
    return disparity_final

def precision(num,pre):
    base=(2**pre)
    binary=bin(int(num*base))[2:]
    integer=int(binary,2)
    result=integer/base
    return result

def downsample(image,height_new,width_new,cut_height,cut_width):
    rows,cols=image.shape
    width_o=cols-cut_width
    height_o=rows-cut_height
    resolution_width=precision(width_o/width_new,17)
    resolution_height=precision(height_o/height_new,17)
    image_cut=image[cut_height:,cut_width:]
    output=np.zeros((height_new,width_new),dtype=np.uint16)
    index_row=0
    index_col=0
    row_new=0
    col_new=0
    for row in range(height_o):
        if (row-1) <= index_row <= row:
            wx0=(index_row-row) % 1
            wx1=1-wx0
            row_d=max(row-1,0)
            index_row+=resolution_height
            for col in range(width_o):
                if ((col-1) <= index_col <= (col)):
                    wy0=(index_col-col+1) % 1
                    wy1=1-wy0
                    w00=wx1*wy1
                    w10=wx0*wy1
                    w01=wx1*wy0
                    w11=wx0*wy0
                    col_d=max(col-1,0)
                    a00=image_cut[row_d,col_d]
                    a01=image_cut[row_d,col]
                    a10=image_cut[row,col_d]
                    a11=image_cut[row,col]
                    summ=a00*w00+a01*w01+a10*w10+a11*w11
                    output[row_new,col_new]=round(summ)
                    col_new+=1
                    index_col+=resolution_width
            index_col=0
            col_new=0
            row_new+=1 
    return output 

@jit(nopython=True)
def hamming_trans_all2(array):
    rows,cols=array.shape
    results=np.zeros_like(array,dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            index=hamming_trans(array[row,col])
            results[row,col]=index
    return results
     
@jit(nopython=True)
def filling3(disparity_map, occlusion, mismatches):
    row, col = disparity_map.shape
    disparity_final=np.zeros((row,col),dtype=np.float32)
    directions = [
        (-1, 0),  # left
        (-1, -1),  # top-left diagonal
        (0, -1),  # top
        (1, -1),  # top-right diagonal
        (1, 0)  # right
    ]

    for j in range(row):
        for i in range(col):
            if occlusion[j, i] :               
                disparity_final[j, i] = disparity_map[j, i]
            elif mismatches[j, i]:
                values = []
                for dx, dy in directions:
                    y, x = j+dy, i+dx
                    while 0 <= y < row and 0 <= x < col:
                        if not mismatches[y, x]  and not occlusion[y, x] :
                            values.append(disparity_map[y,x])
                            break
                        y += dy
                        x += dx
                        
                length=len(values)   
                
                if length==0:
                    disparity_final[j, i] = disparity_map[j, i]
                elif length==1:
                    disparity_final[j, i] = values[0]
                elif length==2:
                    disparity_final[j, i] = min(values)
                elif length==3:
                    disparity_final[j, i] = sorted(values)[1]
                else:
                    disparity_final[j, i] = sorted(values)[2]  # Median value

            else:
                disparity_final[j, i] = disparity_map[j, i]
    print("filling finish")
    return disparity_final
@jit(nopython=True)
def left_right_check(disparity_map, disparity_map_r):
    row, col = disparity_map.shape
    occlusion=np.zeros((row, col),dtype=np.bool_)
    mismatches=np.zeros((row, col),dtype=np.bool_)
    disparity_map_hole=np.zeros((row, col),dtype=np.float32)
    for j in range(row):
        for i in range(col):
            dL = disparity_map[j, i]
            #dL = disparity_map[j, i]/16
            index=int(i-dL)
            if index >= 0 and index < col:
                dR=disparity_map_r[j, index]
                #dR=disparity_map_r[j, index]/16
                value=abs(dL-dR)
                
                # Update output matrices based on whether the total sum exceeds the threshold
                if value > 3:
                    if ((dL-dR) > 0):
                        mismatches[j, i] = True
                    else:
                        occlusion[j, i] = True
                    disparity_map_hole[j, i] = 0
                else:
                    occlusion[j, i] = False
                    mismatches[j, i] = False
                    disparity_map_hole[j, i] = disparity_map[j, i]   # *2 for new filling
            else:
                mismatches[j, i] = True
    print("disparity left_right_check finish")
    return occlusion,mismatches,disparity_map_hole   
@jit(nopython=True)
def filling4(disparity_map, occlusion, mismatches):
    row, col = disparity_map.shape
    disparity_final=np.zeros((row,col),dtype=np.float32)
    directions = [
        (-1, 0),  # left
        (-1, -1),  # top-left diagonal
        (0, -1),  # top
        (1, -1),  # top-right diagonal
        (1, 0)  # right
    ]

    for j in range(row):
        for i in range(col):
            if occlusion[j, i] :
                values = np.zeros(5,dtype=np.float32)
                index=0
                for dx, dy in directions:
                    y, x = j+dy, i+dx
                    while 0 <= y < row and 0 <= x < col:
                        if not mismatches[y, x] and not occlusion[y, x]:
                            values[index]=disparity_map[y, x]
                            index += 1
                            break
                        y += dy
                        x += dx

                if index==0:
                    disparity_final[j,i]=disparity_map[j,i]
                elif index == 1 :
                    disparity_final[j, i] =values[0]
                else:
                    disparity_final[j, i] = sorted(values)[1]  # Second smallest value

            elif mismatches[j, i]:
                values = []
                for dx, dy in directions:
                    y, x = j+dy, i+dx
                    while 0 <= y < row and 0 <= x < col:
                        if not mismatches[y, x]  and not occlusion[y, x] :
                            values.append(disparity_map[y,x])
                            break
                        y += dy
                        x += dx
                        
                length=len(values)   
                
                if length==0:
                    disparity_final[j,i]=disparity_map[j,i]
                elif length==1:
                    disparity_final[j, i] = values[0]
                elif length==2:
                    disparity_final[j, i] = max(values)
                elif length==3:
                    disparity_final[j, i] = sorted(values)[1]
                else:
                    disparity_final[j, i] = sorted(values)[2]  # Median value

            else:
                disparity_final[j, i] = disparity_map[j, i]
    print("filling finish")
    return disparity_final 
def SGM(left_image_path,right_image_path,output_path,disparity_range=256,Grad=32,Ctg=16,Window_Size=3,LARGE_PENALTY=1,SMALL_PENALTY=0.3):
    
    left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)
    if left_image is None or right_image is None:
        raise FileNotFoundError("Could not open or find the images.")
        
    
    left_image_blur = cv2.blur(left_image, (3,3))
    right_image_blur = cv2.blur(right_image, (3,3))
    
    
    left_grad_x, left_grad_y, gradient_left = compute_gradient(left_image_blur)
    right_grad_x, right_grad_y, gradient_right = compute_gradient(right_image_blur)
    
    # start_time=time.time()
    index_left, cost_volume_left, index_right, cost_volume_right = calculate_pixel_cost_all(gradient_left, gradient_right,
                              left_grad_x, left_grad_y, right_grad_x, right_grad_y,
                              disparity_range, Grad, Ctg, Window_Size)
    # end_time=time.time()
    # exe_time=end_time-start_time
    # print(f"time:{exe_time}")
    
    rows,cols=left_image.shape
    print("Aggregating costs for left......")
    aggregated_cost0 = aggregate_costs_0(rows, cols, disparity_range, cost_volume_left, LARGE_PENALTY,  SMALL_PENALTY)
    aggregated_cost135 = aggregate_costs_135(rows, cols, disparity_range, cost_volume_left, LARGE_PENALTY,  SMALL_PENALTY)
    aggregated_cost_left=aggregated_cost0+aggregated_cost135
    # print("Aggregating costs for right......")
    aggregated_cost0_r = aggregate_costs_0(rows, cols, disparity_range, cost_volume_right, LARGE_PENALTY,  SMALL_PENALTY)
    aggregated_cost135_r = aggregate_costs_135(rows, cols, disparity_range, cost_volume_right, LARGE_PENALTY,  SMALL_PENALTY)
    aggregated_cost_right=aggregated_cost0_r+aggregated_cost135_r
    
    # print("Computing disparity.........")

    disparity_map = compute_disparity(aggregated_cost_left, rows, cols, disparity_range,index_left)
    disparity_map_r = compute_disparity(aggregated_cost_right, rows, cols, disparity_range, index_right)
    

    occlusion, mismatches, disparity_map_hole=left_right_check_window3(disparity_map, disparity_map_r, 5)
    # occlusion, mismatches, disparity_map_hole=left_right_check(disparity_map, disparity_map_r)
    # cv2.imwrite(output_path, np.uint16(disparity_map_hole*256))
    
    
    disparity_map_hole16=np.uint16(disparity_map_hole*256)
    num=left_image_path[-13:-4]
    # num="0"
    # folder=output_path+'/no_filling_'+num+'.png'
    folder=output_path+'/no_filling/'+num+'.png'
    cv2.imwrite(folder, disparity_map_hole16)
    # folder3=output_path+'/no_filling_blur_'+num+'.png'
    folder3=output_path+'/no_filling_blur/'+num+'.png'
    disparity_map_hole_median=cv2.medianBlur(disparity_map_hole16, 3)
    cv2.imwrite(folder3, disparity_map_hole_median)
    
    # disparity_final=filling3(disparity_map_hole, occlusion, mismatches)
    disparity_final=filling4(disparity_map_hole, occlusion, mismatches)
    # disparity_final=filling2(disparity_map, disparity_map_r, occlusion, mismatches)
    # folder1=output_path+'/filling_'+num+'.png'
    folder1=output_path+'/filling/'+num+'.png'
    cv2.imwrite(folder1, np.uint16(disparity_final*256))
    disparity_final_blur = median_filter(disparity_final, size=11)
    # folder2=output_path+'/filling_blur_'+num+'.png'
    folder2=output_path+'/filling_blur/'+num+'.png'
    cv2.imwrite(folder2, np.uint16(disparity_final_blur*256))
    
    #disparity_final=filling2(disparity_map, disparity_map_r, occlusion, mismatches)
    # disparity_final2=filling_window2(disparity_map, disparity_map_r, occlusion, mismatches,3)
    
    #disparity_final=np.uint16(disparity_final)
    # disparity_final2=np.uint16(disparity_final2)
    
    #cv2.imwrite('E:/stereokitti/results/filling/000044_10.png', disparity_final*256)
    # cv2.imwrite('E:/stereokitti/results/filling_window/000000_10.png', disparity_final2*256)
    
    #disparity_final_blur = median_filter(disparity_final, size=11)
    # disparity_final_blur2 = median_filter(disparity_final2, size=11)
   
    #cv2.imwrite('E:/stereokitti/results/filling_blur/000044_10.png', disparity_final_blur*256)
    # cv2.imwrite('E:/stereokitti/results/filling_window_blur/000000_10.png', disparity_final_blur2*256)
    
    # print("Downsampling.........")
    # down=downsample(disparity_final_blur*256, 300, 640, 50, 50)
    
    #cv2.imshow("Disparity Map", np.uint8(disparity_map_hole))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
def main():
    # 读取图像
    # left_image_path = 'E:/vkitti/Scene01/15-deg-left/frames/rgb/Camera_0/rgb_00000.jpg'
    # right_image_path = 'E:/vkitti/Scene01/15-deg-left/frames/rgb/Camera_1/rgb_00000.jpg'
    # output_folder = 'E:/vkitti/results/no_filling/Scene01/15-deg-left'
    #output_path = 'E:/vkitti/results/no_filling/Scene01/15-deg-left/rgb_00000.png'
    # filename='rgb_00000.jpg'
    # output_path = os.path.join(output_folder, filename[:-4]+'.png')
    left_image_path = 'D:/stereokitti/data_scene_flow/training/image_2/000104_10.png'
    right_image_path = 'D:/stereokitti/data_scene_flow/training/image_3/000104_10.png'
    # left_image_path = 'D:/Project/SGM/test/000000_10.bmp'
    # right_image_path = 'D:/Project/SGM/test/000000_11.bmp'
    output_path = 'D:/stereokitti/resultsB/'
    start_time=time.time() 
    SGM(left_image_path,right_image_path,output_path)
    end_time=time.time()
    exe_time=end_time-start_time
    print(f"total time: {exe_time}")
if __name__=="__main__":
    main()
