import tracemalloc

# 1. Start tracking
tracemalloc.start()

# 2. Take a snapshot before the import
snapshot1 = tracemalloc.take_snapshot()

# 3. Perform the "heavy" import

# 4. Take a snapshot after
snapshot2 = tracemalloc.take_snapshot()

# 5. Compare the two
stats = snapshot2.compare_to(snapshot1, "lineno")
for stat in stats[:5]:  # Show the top 5 memory hogs
    print(stat)
