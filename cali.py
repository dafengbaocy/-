import cv2
import numpy as np

def cali():
    #stereoParams.CameraParameters1.IntrinsicMatrix
    #ans'
    right_camera_matrix = np.array([[391.257219654230	,0	,0],
                                    [0	,406.580045468676	,0],
                                    [309.707915988516	,335.234852568248	,1]])


    #stereoParams.CameraParameters1
    #顺序：RadialDistortion[0],RadialDistortion[1],TangentialDistortion[0],TangentialDistortion[1],RadialDistortion[2]
    right_distortion = np.array([[0.166593351181639	,-0.0733463431196052,0,0 ,0]])


    #使用时，需要注意参数的排放顺序，即K1，K2，P1，P2，K3。切记不可弄错，否则后续的立体匹配会出现很大的偏差。
    #stereoParams.CameraParameters2.IntrinsicMatrix
    #ans'
    left_camera_matrix = np.array([[390.503853491034	,0	,0],
                                   [0	,404.678788276339	,0],
                                   [289.990075453933	,349.471953297249	,1]])
    #stereoParams.CameraParameters2

    left_distortion = np.array([[0.0632119813549045	,0.0257479023303473,0,0 ,0 ]])
    #stereoParams
    R = np.matrix([
        [ 0.999874121978028	,-0.00429391796355755	,-0.0152742419514269],
        [  0.00469483874327007	,0.999642810434609	,0.0263098847502346],
        [ 0.0151558136648344	,-0.0263782830168716	,0.999537136627368],
    ])


    # print(R)

    T = np.array([-58.2497696583549	,0.800135215555190	,-0.843897191856397]) # 平移关系向量

    size = (640, 480) # 图像尺寸

    # 进行立体更正
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                      right_camera_matrix, right_distortion, size, R,
                                                                      T)
    # 计算更正map
    left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)

    #print(Q)
    return left_map1,left_map2 , right_map1,right_map2,Q

