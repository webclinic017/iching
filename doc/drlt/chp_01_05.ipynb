{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "spectacular-blogger",
   "metadata": {},
   "source": [
    "# Actor-Critic算法\n",
    "近年来在Policy Gradient算法基础上，Actor-Critic算法是最火的一类算法，尤其是A2C（Advantage Actor-Critic）和A3C（Asynchronous Advantage Actor-Critic）算法，以及Soft Actor-Critic算法。\n",
    "## 概述\n",
    "在Actor-Critic中有两个组成部分，Actor代表策略$\\pi _{\\boldsymbol{\\theta}}$，可以根据当前环境状态$S_{t}$，选择合适的行动$A_{t}$，而Critic是一个评判者，会评判看以当前环境状态$S_{t}$，采取行动$A_{t}$的好坏程度。\n",
    "## Actor\n",
    "Actor部分的作用是看到环境状态$S_{t}$，根据当前的策略$\\pi _{\\theta ^{n}}$，选择对应的行动$A_{t}$。为了实现这一目的，可能采用两种方式：\n",
    "* On-Policy：与环境互动的Agent与要学习的Agent是同一个，优点是学习样本都是由当前策略产生的，因此会更有利于学习，缺点是每次策略更新后，就需要重新收集与环境互动的数据，需要耗费大量的时间；\n",
    "* Off-Policy：固定用一个Agent来收集与环境交互的数据，然后拿这些数据来训练另外一个Agent。这种方法的优点是不需要反复与环境互动收集数据，缺点是由于与环境互动的Agent与欲训练的Agent不同，我们需要知道二者的差别，只有这样才能训练好欲训练的Agent，具体训练方法中，通常使用的是PPO算法（Proximal Policy Optimization）；\n",
    "### On-Policy\n",
    "我们采用如下所示的网络结构：\n",
    "![Actor网络架构](./images/chp_01_05_01.png)\n",
    "如图所示，左边为环境状态$S_{t}$，将其输入策略网络$\\pi _{\\theta ^{i}}$，最后的输出层每个神经元代表选择某个行动的概率。\n",
    "#### 基础算法\n",
    "在讲解具体算法之前，我们先来看Agent与环境互动获取训练用数据集的方法。我们利用随机数初始化上面的神经网络，然后拿这个策略神经网络$\\pi _{\\theta ^{0}}$与环境互动$N^{0}$次，对于其中的某个Trajectory为：\n",
    "$$\n",
    "\\tau ^{n}: \\{ (s_{1}^{n}, a_{1}^{n}), (r_{2}^{n}, s_{2}^{n}, a_{2}^{n}), ..., (r_{T_{0}}^{n}, s_{T_{0}}, a_{T_{0}}^{n}) \\}\n",
    "$$\n",
    "对于环境状态为$s_{1}$时，会采取行动$a_{1}$，在其后所获得的累积奖励为：\n",
    "$$\n",
    "G_{1}^{n} = r_{2}^{n} + \\gamma r_{3}^{n} + ... + \\gamma ^{T_{0}-1 -1} r_{T_{0}}^{n} = \\sum _{k=0}^{T_{0}-1 -1} \\gamma ^{k} r_{1+k+1}^{n}\n",
    "$$\n",
    "对于任意时刻$t$累积奖励为：\n",
    "$$\n",
    "G_{t}^{n} = r_{t+1}^{n} + \\gamma r_{t+2}^{n} + ... + \\gamma ^{T_{0}-t-1} r_{T_{0}}^{n} = \\sum _{k=0}^{T_{0}-t-1} \\gamma ^{k} r_{t+k+1}^{n}\n",
    "$$\n",
    "由此我们可以得到如下数据集：\n",
    "$$\n",
    "(s_{1}^{n}, a_{1}^{n}) \\otimes G_{1}^{n} \\\\\n",
    "(s_{2}^{n}, a_{2}^{n}) \\otimes G_{2}^{n} \\\\\n",
    "...... \\\\\n",
    "(s_{i}^{n}, a_{i}^{n}) \\otimes G_{i}^{n} \\\\\n",
    "...... \\\\\n",
    "(s_{T_{0}}^{n}, a_{T_{0}}^{n}) \\otimes G_{T_{0}}^{n} \\\\\n",
    "$$\n",
    "其中对于某个样本数据，$s_{i}^{n}$为网络的输入信号，$a_{i}^{n}$为自然数，代表某个行动的编号，即为一个典型的多分类问题的数据集格式，而$G_{i}^{n}$表明在$s_{i}^{n}$的情况下选择行动$a_{i}^{n}$是好还是不好的程度。\n",
    "算法如下所示：\n",
    "1. 用随机数$\\theta ^{0}$初始化策略网络$\\pi _{\\pi ^{0}}$；\n",
    "2. 循环i=0到I次：\n",
    "  1. 拿$\\pi _{\\theta ^{i}}$网络与环境进行N次互动，得到如下数据集：\n",
    "$$\n",
    "(s_{1}^{1,i}, a_{1}^{1,i}) \\otimes G_{1}^{1,i} \\\\\n",
    "(s_{2}^{1,i}, a_{2}^{1,i}) \\otimes G_{2}^{1,i} \\\\\n",
    "...... \\\\\n",
    "(s_{T_{0}}^{1,i}, a_{T_{0}}^{1,i}) \\otimes G_{T_{0}}^{1,i} \\\\\n",
    "...... \\\\\n",
    "(s_{T_{N}}^{N,i}, a_{T_{N}}^{N,i}) \\otimes G_{T_{N}}^{N,i}\n",
    "$$\n",
    "  2. 利用CrossEntropy算法求出代价函数值$ L(\\theta)$；\n",
    "  3. 更新参数：$\\theta ^{i+1} = \\theta ^{i} - \\alpha \\nabla _{\\theta} L(\\theta)$；\n",
    "  4. 循环2、3步多次，直到达到足够高的精度，然后回到i循环下一个i值；\n",
    "#### Baseline\n",
    "\n",
    "#### Advantage函数\n",
    "\n",
    "### Off-Policy(PPO)\n",
    "## Critic\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "patient-burst",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "caring-engineer",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
