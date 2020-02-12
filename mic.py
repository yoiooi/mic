from minepy import MINE
from scipy.stats import pearsonr, spearmanr
import nlcor
import numpy as np
import matplotlib.pyplot as plt
import time

mine = MINE()

n = 10000
m = 100 # 1% noise
start = time.time() # 시작시간 저장

# x, y는 1차원 실수형 array
"""
random하게 데이터 생성 후 상관관계 확인
참고 - https://datascienceschool.net/view-notebook/ff367da95afc43ed8ae6ec30efc0fb9f/
"""
plt.figure(figsize=(8, 6))

#plt.subplot(231)
x1 = np.random.uniform(-50, 50, n)
#x1 = np.random.uniform(-50, 50, n)
y1 = 2*x1**2 + np.random.uniform(-50, 50, n)
#plt.scatter(x1, y1)
#mine.compute_score(x1, y1)
#print("random - x1, y1", mine.mic())
#plt.title("MIC={0:0.3f}".format(mine.mic()))

#plt.subplot(232)
x2 = np.random.uniform(-50, 50, n)
y2 = 4*(x2**2-1250)**2 + np.random.uniform(-50, 50, n)/5
#plt.scatter(x2, y2)
#mine.compute_score(x2, y2)
#print("random - x2, y2", mine.mic())
#plt.title("MIC={0:0.3f}".format(mine.mic()))

#plt.subplot(233)
x3 = np.random.uniform(-1, 1, n)
y3 = np.cos(x3 * np.pi) + np.random.uniform(0, 1/8, n)
x3 = np.sin(x3 * np.pi) + np.random.uniform(0, 1/8, n)
#plt.scatter(x3, y3)
#mine.compute_score(x3, y3)
#print("random - x3, y3", mine.mic())
#plt.title("MIC={0:0.3f}".format(mine.mic()))

#plt.subplot(234)
x4 = np.random.uniform(-1, 1, n)
y4 = np.random.uniform(-1, 1, n)
#plt.scatter(x4, y4)
#mine.compute_score(x4, y4)
#print("random - x4, y4", mine.mic())
#plt.title("MIC={0:0.3f}".format(mine.mic()))

#plt.subplot(235)
x5 = np.random.uniform(-1, 1, n)
y5 = (x5**2 + np.random.uniform(0, 0.5, n)) * np.array([-1, 1])[np.random.random_integers(0, 1, size=n)]
#plt.scatter(x5, y5)
#mine.compute_score(x5, y5)
#print("random - x5, y5", mine.mic())
#plt.title("MIC={0:0.3f}".format(mine.mic()))

#plt.subplot(236)
xy1 = np.random.multivariate_normal([3, 3], [[1, 0], [0, 1]], int(n/4))
xy2 = np.random.multivariate_normal([-3, 3], [[1, 0], [0, 1]], int(n/4))
xy3 = np.random.multivariate_normal([-3, -3], [[1, 0], [0, 1]], int(n/4))
xy4 = np.random.multivariate_normal([3, -3], [[1, 0], [0, 1]], int(n/4))
xy = np.concatenate((xy1, xy2, xy3, xy4), axis=0)
x6 = xy[:, 0]
y6 = xy[:, 1]
#plt.scatter(x6, y6)
#mine.compute_score(x6, y6)
#print("random - x6, y6", mine.mic())
#plt.title("MIC={0:0.3f}".format(mine.mic()))

#plt.tight_layout()
#plt.show()

#print("random data :", n, " workingTime : {} sec".format(time.time() - start)) # random data 실행시간

"""
linear Test
"""
print("linear Performance compare MIC, Pearson ")
"""
xl1 = 20 * np.random.randn(n) + 100
yl1 = xl1 + (10 * np.random.randn(n) + 50)

xlm1 = np.random.uniform(0, 200, m)
ylm1 = np.random.uniform(0, 250, m)

xxl1 = np.concatenate((xl1,xlm1), axis=0)
yyl1 = np.concatenate((yl1,ylm1), axis=0)

startPearson = time.time() # 시작시간 저장
plt.subplot(231)
plt.scatter(xl1, yl1)
corr, p_value = pearsonr(xl1, yl1)
print("Pearson&linear - xl1, yl1", corr)
pearsonTime = time.time() - startPearson
pearsonTime_min = int(pearsonTime / 60)
pearsonTime_sec = pearsonTime % 60

plt.title("Pearson\n Correlation Coefficient={0:0.3f} \n Computation Time={1}min {2:0.3f}sec".format(corr, pearsonTime_min, pearsonTime_sec))

print("random data :", n, " Pearson&linear workingTime : {} sec".format(time.time() - startPearson)) # 실행시간

startSpearman = time.time() # 시작시간 저장
plt.subplot(232)
plt.scatter(xl1, yl1)
rho, p_value = spearmanr(xl1, yl1, axis=None)
print("SPearman&linear - xl1, yl1", rho)
spearmanTime = time.time() - startSpearman
spearmanTime_min = int(spearmanTime / 60)
spearmanTime_sec = spearmanTime % 60

plt.title("Spearman\n Correlation Coefficient={0:0.3f} \n Computation Time={1}min {2:0.3f}sec".format(rho, spearmanTime_min, spearmanTime_sec))

print("random data :", n, " SPearman&linear workingTime : {} sec".format(time.time() - startSpearman)) # 실행시간

startMIC = time.time() # 시작시간 저장
plt.subplot(233)
plt.scatter(xl1, yl1)
mine.compute_score(xl1, yl1)
print("MIC&linear - xl1, yl1", mine.mic())
MICTime = time.time() - startMIC
MCITime_min = int(MICTime / 60)
MICTime_sec = MICTime % 60

plt.title("MIC\n Correlation Coefficient={0:0.3f} \n Computation Time={1}min {2:0.3f}sec".format(mine.mic(), MCITime_min, MICTime_sec))

print("random data :", n, " MIC&linear workingTime : {} sec".format(time.time() - startMIC)) # 실행시간

plt.tight_layout()
plt.show()
"""
print("non-linear Test")
"""
xm1 = np.random.uniform(-50, 50, m)
ym1 = np.random.uniform(-50, 5000, m)

xx1 = np.concatenate((x1,xm1), axis=0)
yy1 = np.concatenate((y1,ym1), axis=0)

xm2 = np.random.uniform(-50, 50, m)
ym2 = np.random.uniform(-10, 6000000, m)

xx2 = np.concatenate((x2,xm2), axis=0)
yy2 = np.concatenate((y2,ym2), axis=0)

startSpearman = time.time() # 시작시간 저장
plt.subplot(231)
plt.scatter(xx1, yy1)
rho, p_value = spearmanr(xx1, yy1, axis=None)
print("SPearman&linear - xx1, yy1", rho)
spearmanTime = time.time() - startSpearman
spearmanTime_min = int(spearmanTime / 60)
spearmanTime_sec = spearmanTime % 60

plt.title("Spearman\n Correlation Coefficient={0:0.3f} \n Computation Time={1}min {2:0.3f}sec".format(rho, spearmanTime_min, spearmanTime_sec))
print("random data :", n, " SPearman&nonlinear workingTime : {} sec".format(time.time() - startSpearman)) # 실행시간

startMIC = time.time() # 시작시간 저장
plt.subplot(232)
plt.scatter(xx1, yy1)
mine.compute_score(xx1, yy1)
print("MIC&linear - xx1, yy1", mine.mic())
MICTime = time.time() - startMIC
MCITime_min = int(MICTime / 60)
MICTime_sec = MICTime % 60

plt.title("MIC\n Correlation Coefficient={0:0.3f} \n Computation Time={1}min {2:0.3f}sec".format(mine.mic(), MCITime_min, MICTime_sec))
print("random data :", n, " MIC&nonlinear workingTime : {} sec".format(time.time() - startMIC)) # 실행시간

startSpearman = time.time() # 시작시간 저장
plt.subplot(234)
plt.scatter(xx2, yy2)
rho, p_value = spearmanr(xx2, yy2, axis=None)
print("SPearman&linear - xx2, yy2", rho)
spearmanTime = time.time() - startSpearman
spearmanTime_min = int(spearmanTime / 60)
spearmanTime_sec = spearmanTime % 60

plt.title("Spearman\n Correlation Coefficient={0:0.3f} \n Computation Time={1}min {2:0.3f}sec".format(rho, spearmanTime_min, spearmanTime_sec))
print("random data :", n, " SPearman&nonlinear workingTime : {} sec".format(time.time() - startSpearman)) # 실행시간

startMIC = time.time() # 시작시간 저장
plt.subplot(235)
plt.scatter(xx2, yy2)
mine.compute_score(xx2, yy2)
print("MIC&linear - xx2, yy2", mine.mic())
MICTime = time.time() - startMIC
MCITime_min = int(MICTime / 60)
MICTime_sec = MICTime % 60

plt.title("MIC\n Correlation Coefficient={0:0.3f} \n Computation Time={1}min {2:0.3f}sec".format(mine.mic(), MCITime_min, MICTime_sec))
print("random data :", n, " MIC&nonlinear workingTime : {} sec".format(time.time() - startMIC)) # 실행시간

plt.tight_layout()
plt.show()
"""
print("nlcor and MIC comparision")

start_nlcor = time.time() # 시작시간 저장
plt.subplot(231)
plt.scatter(x1, y1)
c = nlcor(x1, y1, plt=T)

nlcorTime = time.time() - start_nlcor
nlcorTime_min = int(nlcorTime / 60)
nlcorTime_sec = nlcorTime % 60

plt.title("nlcor\n nlcor Coefficient={0:0.3f} \n Computation Time={1}min {2:0.3f}sec".format(c.estimate, nlcorTime_min, nlcorTime_sec))
print("random data :", n, " nlcor workingTime : {} sec".format(time.time() - start_nlcor)) # 실행시간

startMIC = time.time() # 시작시간 저장
plt.subplot(232)
plt.scatter(x1, y1)
mine.compute_score(x1, y1)
print("MIC&linear - x1, y1", mine.mic())
MICTime = time.time() - startMIC
MCITime_min = int(MICTime / 60)
MICTime_sec = MICTime % 60

plt.title("MIC\n Correlation Coefficient={0:0.3f} \n Computation Time={1}min {2:0.3f}sec".format(mine.mic(), MCITime_min, MICTime_sec))
print("random data :", n, " MIC&nonlinear workingTime : {} sec".format(time.time() - startMIC)) # 실행시간

plt.tight_layout()
plt.show()