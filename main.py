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
  print 'N = {0}'.format(N)
  print '-' * 80
  print 'x = {0}'.format(x)
  print '-' * 80
  print 'y = {0}'.format(y)
  print '-' * 80
  print 'theta() = {0}'.format(theta())
  print '-' * 80
  print 'f_theta() = {0}'.format(f_theta())
  print '-' * 80
  print 'j_theta() = {0}'.format(j_theta())
  print '-' * 80
  print 'gradient_descent_batch() = {0}'.format(gradient_descent_batch())
  print '-' * 80

def theta():
  return numpy.dot(numpy.linalg.inv(numpy.dot(x, x.T)), numpy.dot(x, y))

def f_theta():
  return numpy.dot(theta().T, x)

def j_theta():
  tmp = (y - numpy.dot(x.T, theta()))
  return ((1.0/N) * numpy.dot(tmp.T, tmp))
  
def gradient_descent_batch():
  xres = numpy.zeros(N)
  
  for i in range(1,75):
    xres = xres - ((1.0/(1000*i)) * j_theta()[xres])
  
  return xres

def print_graphs():
  pyplot.plot(t, p, '.')
  pyplot.plot(t, f_theta())
  pyplot.ylabel('position (m)')
  pyplot.xlabel('temps (s)')
  pyplot.show()
  pyplot.plot(range(0,N), gradient_descent_batch())
  pyplot.show()

def main():
  load_data()
  print_results()
  print_graphs()

if __name__ == '__main__':
  main()
