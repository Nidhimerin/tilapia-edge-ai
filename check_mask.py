import pandas as pd
df = pd.read_csv("results/fixed/pi_feeding_test_signal_with_zscores.csv")
base = df.time_s < 600
win = (df.time_s >= 670) & (df.time_s <= 790)
print("flagged %  baseline:", round(df.shake[base].mean()*100, 1),
      " 670-790:", round(df.shake[win].mean()*100, 1),
      " after 790:", round(df.shake[df.time_s > 790].mean()*100, 1))
for c in ["diff_near", "flow_near"]:
    z = (df[c] - df.loc[base, c].mean()) / df.loc[base, c].std()
    i = z[win].idxmax()
    print(c, "| unmasked peak z:", round(z[win].max(), 1), "at", df.time_s[i],
          "| masked peak z:", round(df[c + "_z"][win].max(), 1),
          "| mean z 670-700 unmasked:",
          round(z[(df.time_s >= 670) & (df.time_s < 700)].mean(), 2))