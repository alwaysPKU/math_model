# -*- coding: utf-8 -*-
import numpy as np
import cv2
import random


class ModifiedVibe:
    def __init__(self):
        """构造函数"""
        self.video = None  # 矩阵格式存放的每一帧图片
        self.N = 20  # 背景样本个数
        self.R = 20.  # 距离阈值
        self.M = None  # 背景样本
        self.sample_factors = 20.  # 采样因子
        self.scale_factor = 4.  # 尺度因子
        self.min_match = 2.  # 最小匹配个数
        self.k = 5  # 邻域半径
        self.a_inc = 0.2  # 自增适应参数
        self.a_dec = 0.5  # 自减适应参数
        self.prob_count = None  # 帧数
        self.width = None  # 影像宽度
        self.height = None  # 影像高度
        self.random_update_prob = 0.5  # 随机更新概率
        self.D = None  # 存放最小距离
        self.D_N = 25  # D的大小
        self.max_R = 90  # 阈值的上限
        self.min_R = 18  # 阈值的下限

    def read_video(self, file_name):
        """读入视频"""
        cap = cv2.VideoCapture(file_name)
        i = 0
        self.prob_count = int(cap.get(7))  # 帧数
        self.width = int(cap.get(3))  # 影像宽度
        self.height = int(cap.get(4))  # 影像高度

        prob_list = []
        ret, frame = cap.read()
        while ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转换为灰度
            i += 1
            prob_list.append(gray)  # 将帧放入列表中
            ret, frame = cap.read()
        cap.release()
        self.video = np.array(prob_list, np.float32)  # 将图像放入np中
        self.M = self.video[range(self.N), :, :].transpose((1, 2, 0))  # 初始化背景样本
        R = self.R
        self.R = np.ones(shape=(self.height, self.width), dtype=np.float32) * R  # 初始化半径参数
        # 初始化
        self.D = list()
        for i in range(self.height):
            self.D.append(list())
            for j in range(self.width):
                self.D[i].append(list([R, ]))

    def bg_sub(self):
        """删除背景"""
        for prob in range(0, self.prob_count):
            is_match = np.zeros(shape=self.video[prob].shape)
            #  遍历像素点，看是否契合背景
            for i in range(self.height):
                for j in range(self.width):
                    self.match_bg(pos=(prob, i, j), is_match=is_match)
            # 遍历像素点，更新背景样本
            for i in range(self.height):
                for j in range(self.width):
                    if random.uniform(0, 1) < self.random_update_prob:
                        self.M[i, j][random.choice(range(self.N))] = self.video[prob, i, j]
                    if is_match[i][j] == 255:
                        self.update_M(pos=(prob, i, j), is_match=is_match)
            # cv2.imshow('', is_match)
            cv2.imwrite('office/img' + str(prob) + '.jpg', is_match)
            cv2.imwrite('office/orig' + str(prob) + '.jpg', self.video[prob])
            print(prob)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    def match_bg(self, pos, is_match):
        """该点是否契合背景"""
        value = self.video[pos[0], pos[1], pos[2]]
        M_x = self.M[pos[1], pos[2]]
        match_count = 0
        l = list()
        for m in M_x:  # 遍历背景样本

            l.append(abs(value - m))
            if abs(value - m) < self.R[pos[1], pos[2]]:
                match_count += 1
        min_d = np.array(l).mean()
        # 更新R_x
        D_x = self.D[pos[1]][pos[2]]
        D_x.append(min_d)
        if len(D_x) > self.D_N:
            self.D[pos[1]][pos[2]] = D_x[-self.D_N:]
            D_x = self.D[pos[1]][pos[2]]
        av_d = np.array(D_x).mean()
        r_x = self.R[pos[1], pos[2]]
        self.R[pos[1], pos[2]] = (r_x * (1 - self.a_dec) if r_x < av_d * self.scale_factor
                                  else r_x * (1 + self.a_inc))
        self.R[pos[1], pos[2]] = max(self.R[pos[1], pos[2]], self.min_R)
        self.R[pos[1], pos[2]] = min(self.R[pos[1], pos[2]], self.max_R)
        if match_count >= self.min_match:  # 背景是255
            is_match[pos[1], pos[2]] = 255

    def update_M(self, pos, is_match):
        """更新背景样本"""
        n_x = .0
        b_x = .0
        for i in range(pos[1] - self.k, pos[1] + self.k):
            for j in range(pos[2] - self.k, pos[2] + self.k):
                if 0 <= i < self.height and 0 <= j < self.width:
                    n_x += 1
                    if is_match[i, j] == 255:
                        b_x += 1
        ncf = b_x / n_x
        # 若ncf > 0.5，则以 1 / ((1 / (2 x ncf)) * sample_factors) 的概率更新背景样本
        if ncf > 0.5:
            pro = 1 / ((1 / (2 * ncf)) * self.sample_factors)
            if random.uniform(0, 1) < pro:
                self.M[pos[1], pos[2]][random.choice(range(self.N))] = self.video[pos[0], pos[1], pos[2]]


if __name__ == '__main__':
    mdv = ModifiedVibe()
    mdv.read_video('office.mov')
    mdv.bg_sub()
