"""Generate a demo PNG showing the forward pass output."""
import subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Capture vit.py output (suppress numpy warning)
result = subprocess.run(
    ["python", "vit.py"],
    capture_output=True, text=True
)
output = result.stdout.strip()

# Also capture pytest output
pytest_result = subprocess.run(
    ["python", "-m", "pytest", "test_vit.py", "-v", "--tb=no", "-q"],
    capture_output=True, text=True
)
pytest_output = pytest_result.stdout.strip()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#1e1e2e")

for ax, text, title in [
    (axes[0], output, "python vit.py"),
    (axes[1], pytest_output, "pytest test_vit.py -v"),
]:
    ax.set_facecolor("#1e1e2e")
    ax.text(
        0.05, 0.95, text,
        transform=ax.transAxes,
        fontsize=11, verticalalignment="top",
        fontfamily="monospace",
        color="#cdd6f4",
        wrap=False,
    )
    ax.set_title(f"$ {title}", color="#89b4fa", fontsize=12,
                 fontfamily="monospace", pad=10, loc="left")
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_edgecolor("#313244")

fig.suptitle("Mini Vision Transformer — Demo", color="#cba6f7",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("demo/output.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved demo/output.png")
