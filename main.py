# -*- coding: utf8 -*-

import numpy
from matplotlib import pyplot
import random

def load_data():
  global x, y, N, t, p
  t = numpy.loadtxt('data/t.txt')
  N = len(t)
  x = numpy.vstack((t, numpy.ones(N)))
  p = numpy.loadtxt('data/p.txt')
  y = p

def print_results():
  print '-' * 80
  print 'N: \n{0}'.format(N)
  print '-' * 80
  print 'x: \n{0}'.format(x)
  print '-' * 80
  print 'y: \n{0}'.format(y)
  print '-' * 80

def theta():
  return numpy.dot(numpy.linalg.inv(numpy.dot(x, x.T)), numpy.dot(x, y))

def f_theta(theta):
  return numpy.dot(theta.T, x)

def pas_batch(val):
  A = 100.0
  B = 1
  C = 10000
  return ((A/(C + (B * val))))

def pas_stochastique(val):
  A = 0.1
  B = 1
  C = 0.0001
  return ((A/(C + (B * val))))

def j_theta(theta):
  tmp = (y - numpy.dot(x.T, theta))
  # numpy.dot(tmp.T, tmp) = le carré
  return ((1.0/N) * numpy.dot(tmp.T, tmp))
  
#Theta moindres carrees = [1.95293789, 3.59623499]
def batch_gradient_descent():
  theta = [1, 1]
  bf = [theta]

  error = j_theta(theta)

  i = 1
  while (abs(error) < 10e4):
    theta = theta + (pas_batch(i) * (1.0/N) * numpy.dot(x, (y - numpy.dot(x.T, theta))))
    bf.append(theta)
    i+=1
    error = error - j_theta(theta)

  # theta = [1.95293789, 3.59623499]
  print "Batch theta = ", theta
  return bf

def stochastique_gradient_descent():
  theta = numpy.array([1, 1])
  sf = [theta]

  error = j_theta(theta)

  for i in range(0, N-1):
    theta = theta + (pas_stochastique(i) * numpy.dot([[x[0][i]], [1]], (y[i] - numpy.dot(theta.T,[[x[0][i]],[1]]))))
    error = error - j_theta(theta)
    sf.append(theta)

  print "Stochastic theta = ", theta
  return sf

def print_graphs():
  batch_res = batch_gradient_descent()
  stochastique_res = stochastique_gradient_descent()

  pyplot.figure(1)
  pyplot.plot(t, p, '.')
  pyplot.plot(t, f_theta(theta()), label="MC")
  pyplot.plot(t, f_theta(batch_res[-1]), '--', label="BGD")
  pyplot.plot(t, f_theta(stochastique_res[-1]), '--', label="SGD")
  pyplot.legend()
  pyplot.grid(True)
  pyplot.ylabel('position (m)')
  pyplot.xlabel('temps (s)')

  pyplot.figure(2)
  pyplot.title('Batch Gradient Descent')
  pyplot.grid(True)
  pyplot.plot(batch_res)

  pyplot.figure(3)
  pyplot.title('Stochastique Gradient Descent')
  pyplot.grid(True)
  pyplot.plot(stochastique_res)

  pyplot.show()

def main():
  load_data()
  print_results()
  print_graphs()

if __name__ == '__main__':
  main()
  
