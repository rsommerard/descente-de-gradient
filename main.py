# -*- coding: utf8 -*-

import numpy
from matplotlib import pyplot

def load_data():
  """
    Charge les données des fichiers textes.
  """
  global x, y, N, t, p
  t = numpy.loadtxt('data/t.txt')
  N = len(t)
  x = numpy.vstack((t, numpy.ones(N)))
  p = numpy.loadtxt('data/p.txt')
  y = p

def print_results():
  """
    Affiche les résultats.
  """
  print 'FAA - TP2: Descente de gradient'
  print '-' * 80
  print 'Nombre de données: {0}'.format(N)
  print '-' * 80
  print 'Rappel théta moindres carrés: [ 1.95293789  3.59623499]'
  print '-' * 80

def theta():
  """
    Calcul théta pour x et y.
  """
  return numpy.dot(numpy.linalg.inv(numpy.dot(x, x.T)), numpy.dot(x, y))

def f_theta(theta):
  """
    Calcul le y pour les x en fonction de théta.
  """
  return numpy.dot(theta.T, x)

def pas_batch(val):
  """
    Pas de la descente de gradient batch.
  """
  A = 100.0
  B = 1
  C = 10000
  return ((A/(C + (B * val))))

def pas_stochastique(val):
  """
    Pas de la descente de gradient stochastique.
  """
  A = 0.1
  B = 1
  C = 0.0001
  return ((A/(C + (B * val))))

def j_theta(theta):
  """
    Calcul de l'erreur quadratique.
  """
  tmp = (y - numpy.dot(x.T, theta))
  return ((1.0/N) * numpy.dot(tmp.T, tmp))

def batch_gradient_descent():
  """
    Calcul de théta par la méthode de descente de gradient batch.
  """
  theta = [1, 1]
  bf = [theta]

  previous = j_theta(theta)
  current = previous + 1

  i = 1
  while (abs(previous - current) > 10e-6):
    previous = current
    theta = theta + (pas_batch(i) * (1.0/N) * numpy.dot(x, (y - numpy.dot(x.T, theta))))
    bf.append(theta)
    i+=1
    current = j_theta(theta)

  print "Théta batch = ", theta
  print "    > C'est bien le même théta que par la méthode des moindres carrés (même données en entrée)."
  print "    > Sur le graph, on voit bien le théta converger progressivement."
  return bf

def stochastique_gradient_descent():
  """
    Calcul de théta par la méthode de descente de gradient stochastique.
  """
  theta = numpy.array([1, 1])
  sf = [theta]

  error = j_theta(theta)

  for i in range(0, N-1):
    theta = theta + (pas_stochastique(i) * numpy.dot([[x[0][i]], [1]], (y[i] - numpy.dot(theta.T,[[x[0][i]],[1]]))))
    error = error - j_theta(theta)
    sf.append(theta)

  print "Théta stochastique = ", theta
  print "    > Ici encore, le théta se raproche de celui calculé par la méthode des moindres carrés (même données en entrée)."
  print "    > Sur le graph, on voit bien les oscillation (contrairement au batch où c'est progressif)."
  return sf

def print_graphs():
  """
    Affiche les données sur le graph.
  """
  batch_res = batch_gradient_descent()
  stochastique_res = stochastique_gradient_descent()

  print '-' * 80
  print 'Légende graph:'
  print '    - MC: Droite pour la méthode des moindres carrés (TP1)'
  print '    - BGD: Droite pour la méthode de descente de gradient batch'
  print '    - SGD: Droite pour la méthode de descente de gradient stochastique'
  print '-' * 80


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
  pyplot.plot(batch_res, label="theta")

  pyplot.figure(3)
  pyplot.title('Stochastique Gradient Descent')
  pyplot.grid(True)
  pyplot.plot(stochastique_res, label="theta")

  pyplot.show()

def main():
  """
    Fonction principale du programme.
  """
  load_data()
  print_results()
  print_graphs()

if __name__ == '__main__':
  main()
