import pandas as pd
import matplotlib.pyplot as plt

SCOPUS_FILE = "primer_scopus.csv"
df = pd.read_csv(SCOPUS_FILE)


# Plotting the histogram
plt.bar(df['YEAR'], df['Number of papers'])

# Adding labels and title
plt.xlabel('Year')
plt.ylabel('Number of publications')
plt.title('Number of publications per Year')

# Saving the plot to a file (e.g., a PNG file)
plt.savefig("histogram.png")

# Display the plot (optional)
plt.show()

