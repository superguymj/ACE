import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import *


def evaluate(Q, index):
    sum = 0
    for i in index:
        if i > 0:
            sum = sum + Q[i - 1]
        else:
            sum = sum - Q[-i - 1]
    return sum


class BFI_2:
    def __init__(self, Q):
        self.Extraversion = evaluate(
            Q, [1, 6, -11, -16, 21, -26, -31, -36, 41, 46, -51, 56])
        self.Agreeableness = evaluate(
            Q, [2, 7, -12, -17, -22, 27, 32, -37, -42, -47, 52, 57])
        self.Conscientiousness = evaluate(
            Q, [-3, -8, 13, 18, -23, -28, 33, 38, 43, -48, 53, -58])
        self.NegativeEmotionality = evaluate(
            Q, [-4, -9, 14, 19, -24, -29, 34, 39, -44, -49, 54, 59])
        self.Open_Mindedness = evaluate(
            Q, [-5, 10, 15, 20, -25, -30, 35, 40, -45, -50, -55, 60])
        self.vector = [self.Extraversion, self.Agreeableness,
                       self.Conscientiousness, self.NegativeEmotionality, self.Open_Mindedness]
        
    def weight(self, model):
        w0, w1 = 0, 0
        for (v, a0, a1) in zip(self.vector, model.A[0], model.A[1]):
            w0 += v * a0
            w1 += v * a1
        return w0, w1

    def show(self):
        size = 5
        labels = [u'外向性', u'宜人性', u'尽责性', u'负性情绪', u'开放性', u'外向性']
        font = FontProperties(fname=r'C:/Windows/Fonts/simfang.ttf', size=12)

        ax = plt.subplot(221, polar=True)
        # ax.grid(False)

        data = np.array([self.Extraversion, self.Agreeableness,
                        self.Conscientiousness, self.NegativeEmotionality, self.Open_Mindedness])
        data = np.append(data, data[0])

        theta = np.linspace(0, 2*np.pi, size, endpoint=False)
        theta = np.append(theta, theta[0])

        ax.plot(theta, data, 'r')
        ax.fill(theta, data, 'r', alpha=0.3)
        ax.set_xticks(theta)
        ax.set_xticklabels(labels, y=-0.2, fontproperties=font)
        ax.set_yticks([-24, -12, 0, 12, 24])
        ax.yaxis.set_major_formatter(plt.NullFormatter())

        plt.show()
