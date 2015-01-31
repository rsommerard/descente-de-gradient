# -*- coding: utf8 -*-

import numpy
from matplotlib import pyplot

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

def pas(val):
  A = 100.0
  B = 1
  C = 10000
  return ((A/(C + (B * val))))

def j_theta(theta):
  tmp = (y - numpy.dot(x.T, theta))
  # numpy.dot(tmp.T, tmp) = le carré
  return ((1.0/N) * numpy.dot(tmp.T, tmp))
  
#Theta moindres carrees = [1.95293789, 3.59623499]
def batch_gradient_descent():
  theta = [0, 0]
  f = [theta]

  error = j_theta(theta)

  i = 1
  while (abs(error) < 10e4):
    theta = theta + (pas(i) * (1.0/N) * numpy.dot(x, (y - numpy.dot(x.T, theta))))
    f.append(theta)
    i+=1
    error = error - j_theta(theta)

  # theta = [1.95293789, 3.59623499]
  print "theta = ", theta
  return f

def print_graphs():
  pyplot.figure(1)
  pyplot.title('Batch Gradient Descent')
  pyplot.grid(True)
  res = batch_gradient_descent()
  pyplot.plot(res)

  pyplot.figure(2)
  pyplot.plot(t, p, '.')
  pyplot.plot(t, f_theta(theta()))
  pyplot.ylabel('position (m)')
  pyplot.xlabel('temps (s)')
  pyplot.show()

def main():
  load_data()
  print_results()
  print_graphs()

if __name__ == '__main__':
  main()
