from pstats import Stats

stats = Stats("./profile_results/profile.prof")
stats.strip_dirs()
stats.sort_stats('cumulative')
stats.print_callers()
