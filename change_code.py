import scipy as scipy
from numpy.random import uniform, np
import scipy.stats#统计分析包（概率密度）正态分布：norm;poisson 泊松分布

np.set_printoptions(threshold=3)#设置显示3个数字
np.set_printoptions(suppress=True)#使用由科学计数法表示的浮点数
import cv2


def drawLines(img, points, r, g, b):
    cv2.polylines(img, [np.int32(points)], isClosed=False, color=(r, g, b))
#多边形的绘制，非闭合

def drawCross(img, center, r, g, b):
    d = 5
    t = 2
    LINE_AA = cv2.LINE_AA if cv2.__version__[0] == '3' else cv2.CV_AA # cv2.LINE_AA 为抗锯齿,这样看起来会非常平滑
    color = (r, g, b)
    ctrx = center[0, 0]
    ctry = center[0, 1]
    cv2.line(img, (ctrx - d, ctry - d), (ctrx + d, ctry + d), color, t, LINE_AA)
    cv2.line(img, (ctrx + d, ctry - d), (ctrx - d, ctry + d), color, t, LINE_AA)


def mouseCallback(event, x, y, flags, null):
    global center
    global trajectory
    global previous_x
    global previous_y
    global zs
    global po

    center = np.array([[x, y]])#机器人当前位置
    trajectory = np.vstack((trajectory, np.array([x, y])))#将两个方向的数组按垂直方向叠加成一个新的数组
    # noise=sensorSigma * np.random.randn(1,2) + sensorMu

    if previous_x > 0:
        heading = np.arctan2(np.array([y - previous_y]), np.array([previous_x - x]))#返回给定的 X 及 Y 坐标值的反正切值
#射线在原点结束经过点(1,0)与射线在原点结束经过点(x2, x1)之间的弧度符号角。
        if heading > 0:
            heading = -(heading - np.pi)#类似补角计算
        else:
            heading = -(np.pi + heading)

        distance = np.linalg.norm(np.array([[previous_x, previous_y]]) - np.array([[x, y]]), axis=1)
#计算范数，两点之间的距离
        std = np.array([2, 4])
        u = np.array([heading, distance])
        predict(particles, u, std, dt=1.)   #采样
        zs = (np.linalg.norm(landmarks - center, axis=1) + (np.random.randn(NL) * sensor_std_err))#地标各机器人的距离
        #sensor_std_err = 5
        update(particles, weights, z=zs, R=50, landmarks=landmarks)#更新粒子权重

        indexes = systematic_resample(weights)#重新采样后的位置
        resample_from_index(particles, weights, indexes)

        #输出粒子位置
        particle_pos(particles, weights, po)
        #po = np.array(po)

    previous_x = x
    previous_y = y


WIDTH = 800
HEIGHT = 600
WINDOW_NAME = "Particle Filter"

# sensorMu=0
# sensorSigma=3

sensor_std_err = 5


def create_uniform_particles(x_range, y_range, N):#初始化粒子
    particles = np.empty((N, 2))                    #返回N个2为数组，即随机粒子坐标
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)   #x为生x_range[0]~x_range[1]的随机数
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    return particles


def predict(particles, u, std, dt=1.):
    N = len(particles)      #粒子的个数； std=[2,4]
    dist = (u[1] * dt) + (np.random.randn(N) * std[1])
    #u中为角度和距离； np.random.randn：通过本函数可以返回一个或一组服从标准正态分布的随机样本值
    particles[:, 0] += np.cos(u[0]) * dist  #预测粒子的位置
    particles[:, 1] += np.sin(u[0]) * dist


def update(particles, weights, z, R, landmarks):
    weights.fill(1.)        #初始化权重
    for i, landmark in enumerate(landmarks):    #遍历每个路标，且循环处理每个路标
        #np.power:数组元素求n次方； “**"为a的b次方
        distance = np.power((particles[:, 0] - landmark[0]) ** 2 + (particles[:, 1] - landmark[1]) ** 2, 0.5)
        #粒子隔地标的距离
        weights *= scipy.stats.norm(distance, R).pdf(z[i])  #pdf为概率密度，R=50；
        # 均值为distance，方差为50，分布到机器人各坐标的概率； 在位置xt处获得观测量zt的概率p(zt|xt)

    weights += 1.e-300  # avoid round-off to zero
    weights /= sum(weights)     #归一化


def neff(weights):  #采用有效粒子数Neff衡量粒子权值的退化程度
    return 1. / np.sum(np.square(weights))  #权值的平方求和的倒数
#有效粒子数越小，表明权值退化越严重。当Neff的值小于某一阈值时，应当采取重采样措施，根据粒子权值对离散粒子进行重采样。

def systematic_resample(weights):           #重采样
    N = len(weights)
    positions = (np.arange(N) + np.random.random()) / N         #np.arange创建等差数列

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)     #；  cumsum：当前列之前的和加到当前列上
    i, j = 0, 0
    while i < N and j < N:
        if positions[i] < cumulative_sum[j]:        #将粒子移到权重大的地方
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes

#机器人位置的估计值通过粒子的加权平均计算出来，离小车真实值越近的粒子在估计中所占的比重越大。
def estimate(particles, weights):   #计算状态变量估计值
    pos = particles[:, 0:2]
    po = np.average(pos, weights=weights, axis=0)
    var = np.average((pos - po) ** 2, weights=weights, axis=0)
    return po, var


def resample_from_index(particles, weights, indexes):   #根据索引重采样
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights /= np.sum(weights)

def particle_pos(particles, weights, po):
    mo, var = estimate(particles, weights)
    po[0,0] = mo[0]
    po[0,1] = mo[1]

x_range = np.array([0, 800])
y_range = np.array([0, 600])

# Number of partciles
N = 400

landmarks = np.array([[144, 73], [410, 13], [336, 175], [718, 159], [178, 484], [665, 464]])
NL = len(landmarks)
particles = create_uniform_particles(x_range, y_range, N)
po = np.array([[-10, -10]])
weights = np.array([1.0] * N)

# Create a black image, a window and bind the function to window
img = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, mouseCallback)

center = np.array([[-10, -10]])

trajectory = np.zeros(shape=(0, 2)) #轨迹
robot_pos = np.zeros(shape=(0, 2))
previous_x = -1
previous_y = -1
DELAY_MSEC = 50

while (1):

    cv2.imshow(WINDOW_NAME, img)
    img = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    drawLines(img, trajectory, 0, 255, 0)
    drawCross(img, center, r=255, g=0, b=0)

    # landmarks
    for landmark in landmarks:
        cv2.circle(img, tuple(landmark), 10, (255, 0, 0), -1)

    # draw_particles:
    for particle in particles:
        cv2.circle(img, tuple((int(particle[0]), int(particle[1]))), 1, (255, 255, 255), -1)

    # pos
    for p in po:
        cv2.circle(img, tuple(p), 5, (255, 0, 255),-1)
    #po = np.array(po)
    #drawCross(img, po, r=255, g=255, b=0)

    if cv2.waitKey(DELAY_MSEC) & 0xFF == 27:
        break

    cv2.circle(img, (10, 10), 10, (255, 0, 0), -1)
    cv2.circle(img, (10, 30), 3, (255, 255, 255), -1)
    cv2.circle(img, (10, 70), 5, (255, 0, 255), -1)
    cv2.putText(img, "Landmarks", (30, 20), 1, 1.0, (255, 0, 0))
    cv2.putText(img, "Particles", (30, 40), 1, 1.0, (255, 255, 255))
    cv2.putText(img, "Position", (30, 80), 1, 1.0, (255, 0, 255))
    cv2.putText(img, "Robot Trajectory(Ground truth)", (30, 60), 1, 1.0, (0, 255, 0))

    drawLines(img, np.array([[10, 55], [25, 55]]), 0, 255, 0)

cv2.destroyAllWindows()
