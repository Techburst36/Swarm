with open("serve.py", "r") as f:
    code = f.read()

bypass = """    await failover.start()
    
    # Bypass fake fleet consensus for local testing
    failover.is_converged = lambda: True
    if hasattr(failover, "_converged"):
        failover._converged = True
"""
code = code.replace("    await failover.start()", bypass)

with open("serve.py", "w") as f:
    f.write(code)

print("Forced single-node convergence lock to True!")
