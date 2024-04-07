Unsupervised Machine Learning Analysing S&P 500

What has been done:

•Download SP500 stocks prices data.
•Calculate different technical indicators and features for each stock.

•Aggregate on monthly level and filter for each month only top 150 most liquid stocks.
•Calculate monthly returns for different time-horizons to add to features. •Download Fama-French Factors and calculate rolling factor betas for each stock.
•For each month fit a K-means clustering model to group similar assets based on their features.
•For each month select assets based on the cluster and form a portfolio based on Efficient Frontier max sharpe ratio portfolio optimization.
• Visualize the portfolio returns and compare to SP500 returns.

! Limitation! We are going to use most recent SP500 stocks list, which means that there may be a survivorship bias in this list, in reality you have to use survivorship free data.


@https://www.youtube.com/watch?v=9Y3yaoi9rUQ regarding freeCodeCamp.org
