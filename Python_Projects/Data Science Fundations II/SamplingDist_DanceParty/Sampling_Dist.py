import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

def choose_statistic(x, sample_stat_text):
  # calculate mean if the text is "Mean"
  if sample_stat_text == "Mean":
    return np.mean(x)
  # calculate minimum if the text is "Minimum"
  elif sample_stat_text == "Minimum":
    return np.min(x)
  # calculate variance if the text is "Variance"
  elif sample_stat_text == "Variance":
    return np.var(x)
  # if you want to add an extra stat
  elif sample_stat_text == "Median":
    return np.median(x)
  # raise error if sample_stat_text is not "Mean", "Minimum", or "Variance"
  else:
    raise Exception('Make sure to input "Mean", "Minimum", or "Variance"')

def population_distribution(population_data):
  # plot the population distribution
  sns.histplot(population_data, stat='density')
  # informative title for the distribution 
  plt.title(f"Population Distribution")
  #plot the normal distribution curve of the population
  mu_pop = np.mean(population_data)
  sigma_pop = np.std(population_data)
  x_pop = np.linspace(mu_pop - 3*sigma_pop, mu_pop + 3*sigma_pop, 200)
  plt.plot(x_pop, stats.norm.pdf(x_pop, mu_pop, sigma_pop), color='green', label = 'normal PDF')
  # remove None label
  plt.xlabel('')
  plt.show()
  plt.clf()

def sampling_distribution(population_data, samp_size, stat):
  # list that will hold all the sample statistics
  sample_stats = []
  for i in range(500):
    # get a random sample from the population of size samp_size
    samp = np.random.choice(population_data, samp_size, replace = False)
    # calculate the chosen statistic (mean, minimum, or variance) of the sample
    sample_stat = choose_statistic(samp, stat)
    # add sample_stat to the sample_stats list
    sample_stats.append(sample_stat)
  
  pop_statistic = round(choose_statistic(population_data, stat),2)
  # plot the sampling distribution
  sns.histplot(sample_stats, stat='density')
  # informative title for the sampling distribution
  plt.title(f"Sampling Distribution of the {stat} \nMean of the Sample {stat}s: {round(np.mean(sample_stats), 2)} \n Population {stat}: {pop_statistic}")
  plt.axvline(pop_statistic,color='g',linestyle='dashed', label=f'Population {stat}')
  # plot the mean of the chosen sample statistic for the sampling distribution
  plt.axvline(np.mean(sample_stats),color='orange',linestyle='dashed', label=f'Mean of the Sample {stat}s')
  # plot the normal curve on top of the distribution
  mu = np.mean(population_data)
  sigma = np.std(population_data)/(samp_size**.5)
  x = np.linspace(mu - 3*sigma, mu + 3*sigma, 200)
  plt.plot(x, stats.norm.pdf(x, mu, sigma), color='k', label = 'normal PDF')
  plt.legend()
  plt.show()
  plt.clf()

# task 1: load in the spotify dataset
spotify_data = pd.read_csv("genres_v2.csv")
# task 2: preview the dataset
print(spotify_data.head())
# task 3: select the relevant column
song_tempos = spotify_data.tempo
# task 5: plot the population distribution with the mean labeled
population_distribution(song_tempos)
# task 6: sampling distribution of the sample mean
sampling_distribution(song_tempos,50,"Mean")
# task 8: sampling distribution of the sample minimum
sampling_distribution(song_tempos,50,"Minimum")
# task 10: sampling distribution of the sample variance
sampling_distribution(song_tempos,50,"Variance")
# task 13: calculate the population mean and standard deviation
population_mean = song_tempos.mean()
population_std = song_tempos.std()
# task 14: calculate the standard error
standard_error = population_std/(50**0.5)
# task 15: calculate the probability of observing an average tempo of 140bpm or lower from a sample of 30 songs
print(stats.norm.cdf(140,population_mean,standard_error))
# task 16: calculate the probability of observing an average tempo of 150bpm or higher from a sample of 30 songs
print(1- stats.norm.cdf(150,population_mean,standard_error))
# Extra
sampling_distribution(song_tempos,50,"Median")
