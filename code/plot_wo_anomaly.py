"""
==========================================
Sailency : Plotting without anomalies
==========================================

"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.io as sio
import os


#이미 앞에서 뽑아 엑셀로 저장한 값들을 다시 불러와서 플롯팅만 하는 섹션
#그림 세이브도 있으니까 주의해서 실행



def plot_and_anomalypoint(INPUT_FOLDER,filename):

    # 해당파일 리딩
    # 첫번째 로는 인덱스로 판단해서 알아서 안들어갔으니, 첫번째 칼럼은 코딩으로 빼준다.
    # 이런경우가 종종있으니 반드시 읽어오는 파일을 먼저 찍어서 생긴걸 확인하자.
    dist = pd.read_csv(INPUT_FOLDER+filename)
    dist =  dist.iloc[ : , 1: ]
    #dist = dist.head(200)        #실험을 위해 작은사이즈 필요할 경우 활성화

    #확인
    print("##### Input data info #####")
    print("name : ",filename)
    print("shape : ", dist.shape)
    print(" head(5) : \n",dist.iloc[0:5,:])


    # =================================== Plotting ===================================
    # 1. X 값
    # (1) x에 인덱스를 넣는 경우(0,1,2,3,...)
    # 0부터 데이터의 길이까지만큼 간격을 1씩둔 값을 x에 넣음 (len(x) = (43625,2))
    #x = np.arange(0.,len(dist.index),1)
    #print(len(x))

    # (2) x에 시간을 표기할 경우
    # 시작일부터 dist의 row수와 동일한 수의 타임스탬프를 10초간격으로 생성
    x = pd.date_range('2013-05-01', periods=len(dist), freq='10T')
    print("##### timestamp info #####")
    print("num of timestamp : ", len(x))
    print("head 5 : \n", x[0:5])
    print("tail 5 : \n", x[-5: ])

    # 2. Y 값 (거리와 쓰레숄드 2개를 한 차트에 모두 그린다)
    # (1) y : distance
    y = dist.iloc[:,0]

    # (2) y2 : threshold
    n_std = int(filename.split('+')[1].split('std')[0])
    #y2 = dist.iloc[:,1] + pow(dist.iloc[:,2],2)      #지수승 하는 경우
    y2 = dist.iloc[:,1] + (dist.iloc[:,2]*n_std)
    y2_info = "mean+"+str(n_std)+"std"       #그래프에 표기하려고 있는 부분. 이대로 그래프에 나옴. 식바꾸면 여기도 바꿀것.

    # CF
    #y0 = y - y2

    # 3. Plotting
    plt.figure(figsize=(32,4),dpi=600)
    plt.plot(x, y, color='b', label='distance',linewidth=1)
    plt.plot(x, y2, color='r', label='threshold',linewidth=1)
    plt.xlabel("Time")
    plt.ylabel("Distance Values")
    plt.title("Error score and Dynamic thresholding (" + filename.split('.')[0] + ")")
    plt.legend(loc='best')      #그래프안에 라인 이름표

    # 범위지정
    # plt.xlim('2013-09-25', '2013-09-27')
    # plt.ylim(1.2*min(y), 1.2**max(y))


    plt.grid()



    # save plot
    #FIG_FOLDER = r'C:\Users\sunyk\Documents\yj\\research\space\data_from_py\T2\error_score_fig\kmeans\\'
    FIG_FOLDER = r'C:\Users\sunyk\Documents\yj\\research\space\data_from_py\T2\error_score_fig\\'
    plt.savefig(FIG_FOLDER + filename.split('.')[0] + '_plot.png',dpi=600)

    plt.show()


    # =================================== Anomaly points ===================================
    anomaly_t = []
    anomaly_d = []
    anomaly_th = []
    print("Anomaly points ( threshold < error score ) : ")
    for i in range(0,len(y)):
        if y2[i] < y[i]:
            tmp_t = x[i]
            tmp_d = y[i]
            tmp_th = y2[i]
            anomaly_t = np.hstack([anomaly_t,tmp_t])
            anomaly_d = np.hstack([anomaly_d, tmp_d])
            anomaly_th = np.hstack([anomaly_th, tmp_th])

    # 아노멀리정보 데이터 프레임 생성 (서브시스템이름,시간,거리,쓰레숄드)
    df_anomaly = pd.DataFrame(data={'subsys':filename.split('_cl')[0], 'time':anomaly_t, 'dist':anomaly_d,
                                 'threshold':anomaly_th},columns=['subsys','time','dist','threshold'])

    print(" ##### anomaly points #####")
    print("num :",len(df_anomaly))
    print("points : ",df_anomaly)




    # df_anomaly FILE OUT

    # OUTPUT_FOLDER = r'C:\Users\sunyk\Documents\yj\\research\space\data_from_py\T2\threshold\kmeans\\'
    OUTPUT_FOLDER = r'C:\Users\sunyk\Documents\yj\\research\space\data_from_py\T2\anomaly_points\\'
    df_anomaly.to_csv(OUTPUT_FOLDER + filename.split('.')[0] + '_anomaly_points.csv')





# =================================== Input Files ===================================
# 입력파일 변경쉽도록 따로 선언
# 파일 이름 앞에 반드시 'r'을 붙여야 읽힌다: it converts normal string to raw string
INPUT_FOLDER = r'C:\Users\sunyk\Documents\yj\research\space\data_from_py\T2\threshold\kmeans\\'
filename_1 = 'K2_AOCS_CSS_cl_3_thre_w216mean+6std.csv'
filename_2 = 'K2_AOCS_FSS_cl_7_thre_w216mean+6std.csv'
filename_3 = 'K2_AOCS_STA2_cl_2_thre_w200mean+5std.csv'
filename_4 = 'K2_EPS_SAR_cl_9_thre_w100mean+6std.csv'
filename_5 = 'K2_TCS_CENTemp_cl_5_thre_w9mean+5std.csv'
filename_6 = 'K2_TCS_IPTemp_cl_2_thre_w200mean+5std.csv'
filename_7 = 'K2_TCS_LPPTemp_cl_2_thre_w200mean+6std.csv'
filename_8 = 'K2_TCS_SATemp_cl_2_thre_w27mean+5std.csv'
filename_9 = 'K2_TCS_UPPTemp_cl_5_thre_w200mean+4std.csv'

# =================================== COMMAND LINE ===================================


#EXECUTE_LIST = [filename_1,filename_2,filename_3,filename_4,filename_5,filename_6,filename_7,filename_8,filename_9]

#EXECUTE_LIST = [filename_1]
# 실행을 원하는 서브 시스템아래의 모든 파일의 이름(파일내용아니고 이름)을 읽어서 all_files에 리스트로 넣음
all_files = os.listdir(INPUT_FOLDER)
EXECUTE_LIST = [file for file in all_files if file.endswith("w198mean+6std.csv")]
# print(all_files)      #확인용
for filename in EXECUTE_LIST:
    plot_and_anomalypoint(INPUT_FOLDER,filename)

EXECUTE_LIST = [file for file in all_files if file.endswith("w198mean+5std.csv")]
# print(all_files)      #확인용
for filename in EXECUTE_LIST:
    plot_and_anomalypoint(INPUT_FOLDER,filename)

EXECUTE_LIST = [file for file in all_files if file.endswith("w288mean+6std.csv")]
# print(all_files)      #확인용
for filename in EXECUTE_LIST:
    plot_and_anomalypoint(INPUT_FOLDER,filename)

EXECUTE_LIST = [file for file in all_files if file.endswith("w288mean+5std.csv")]
# print(all_files)      #확인용
for filename in EXECUTE_LIST:
    plot_and_anomalypoint(INPUT_FOLDER,filename)

EXECUTE_LIST = [file for file in all_files if file.endswith("w432mean+6std.csv")]
# print(all_files)      #확인용
for filename in EXECUTE_LIST:
    plot_and_anomalypoint(INPUT_FOLDER,filename)

EXECUTE_LIST = [file for file in all_files if file.endswith("w432mean+5std.csv")]
# print(all_files)      #확인용
for filename in EXECUTE_LIST:
    plot_and_anomalypoint(INPUT_FOLDER,filename)

EXECUTE_LIST = [file for file in all_files if file.endswith("w576mean+6std.csv")]
# print(all_files)      #확인용
for filename in EXECUTE_LIST:
    plot_and_anomalypoint(INPUT_FOLDER,filename)

EXECUTE_LIST = [file for file in all_files if file.endswith("w576mean+5std.csv")]
# print(all_files)      #확인용
for filename in EXECUTE_LIST:
    plot_and_anomalypoint(INPUT_FOLDER,filename)






"""

# 이부분은 T1 텐서였을때 사용하였던 데이터 확인용 플롯팅
# =============================== Input Data : Data plotting ===============================

#데이터가 어떻게 생겼는지 파악하기 위해서 전체적으로 데이터값을 플롯팅 해본다
#x에는 그냥 0,1,2,3,... 의 값을 데이터 길이 만큼 넣어주고 y에 데이터 값을 넣어서 찍으면 타임시리즈 형태의 긴 데이터가 나온다

# TCS_UPPTemp[:,4].plot()
# x=AOCS_CSS.iloc[:,0]
# x=np.arange(0.,5.,0.2)

# x에 인덱스를 넣는다(0,1,2,3,...)
# 0부터 데이터의 길이까지만큼 간격을 1씩둔 값을 x에 넣음
x=np.arange(0.,len(dist.index),1)
# print(x.head(10))

# y에 플롯팅하고자 하는 값을 넣음
# ACSS1I 텔레메트리값을 한번 보고 싶다면 6번 칼럼의 모든값 넣음. 여기서 6번은 정확히는 7번째 (ACSS1I). 첫번째로가 0이니까
y=dist.iloc[:,1]

# 이제 플롯팅 한다
# plt.xlim(0, 3)
# print(y.head(10))
plt.plot(x,y)
#plt.xlim(0, 1000)
# plt.figure(num=1,dpi=100,facecolor='white')
# plt.plot(x,y,color='blue',linewidth=1, linestyle='-',label="sampled_AOCS_CSS")
# plt.title('sampled AOCS_CSS')
# plt.xlabel('time')
# plt.ylabel('value')
# plt.xlim(0, 50)
# plt.ylim(1.2*min(y), 1.2**max(y))

# plt.xticks(np.arange(0,5.5,0.5))
plt.grid()
plt.show()
"""