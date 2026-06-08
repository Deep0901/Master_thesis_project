import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

os.makedirs('images', exist_ok=True)

usability = {
    'Interface clarity': 4.5,
    'Query modification': 4.3,
    'Search intuitiveness': 4.4,
    'Layout clarity': 4.2,
    'Feedback usefulness': 4.3,
}
relevance = {
    'Dataset relevance': 4.4,
    'Ranking quality': 4.3,
    'Exploratory query handling': 4.2,
    'Fuzzy query support': 4.3,
}
explainability = {
    'Explanation clarity': 4.4,
    'Transparency': 4.3,
    'Metadata usefulness': 4.5,
    'Trust in results': 4.4,
}
participant = {
    'Familiar with dataset search': 8,
    'Not familiar': 2,
    'Previous OGD experience': 6,
    'No OGD experience': 4,
}

overall = {
    'Usability': sum(usability.values()) / len(usability),
    'Relevance': sum(relevance.values()) / len(relevance),
    'Explainability': sum(explainability.values()) / len(explainability),
}


def save_bar_chart(data, filename, title, xlabel, ylabel, figsize=(10, 6), rotation=15):
    fig, ax = plt.subplots(figsize=figsize)
    labels = list(data.keys())
    values = list(data.values())
    colors = [
        '#0057b7' if v >= 4.4 else '#ffd700' if v >= 4.3 else '#d62828'
        for v in values
    ]
    ax.bar(labels, values, color=colors)
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 5)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xticklabels(labels, rotation=rotation, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)

save_bar_chart(
    usability,
    'images/usability_evaluation_chart.png',
    'Usability Evaluation Results',
    'Usability dimension',
    'Mean score (1–5)',
)
save_bar_chart(
    relevance,
    'images/relevance_evaluation_chart.png',
    'Relevance Evaluation Results',
    'Relevance dimension',
    'Mean score (1–5)',
)
save_bar_chart(
    explainability,
    'images/explainability_evaluation_chart.png',
    'Explainability Evaluation Results',
    'Explainability dimension',
    'Mean score (1–5)',
)

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(list(overall.keys()), list(overall.values()), color=['#0057b7', '#ffd700', '#d62828'])
ax.set_title('Overall Prototype Evaluation Comparison', fontsize=14, pad=12)
ax.set_xlabel('Evaluation category')
ax.set_ylabel('Average score (1–5)')
ax.set_ylim(0, 5)
ax.grid(axis='y', linestyle='--', alpha=0.4)
fig.tight_layout()
fig.savefig('images/overall_evaluation_comparison_chart.png', dpi=300)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 6))
labels = list(participant.keys())
values = list(participant.values())
ax.bar(labels, values, color=['#0057b7', '#ffd700', '#0057b7', '#ffd700'])
ax.set_title('Participant Demographics and Experience', fontsize=14, pad=12)
ax.set_xlabel('Participant group')
ax.set_ylabel('Count')
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.set_xticklabels(labels, rotation=25, ha='right')
fig.tight_layout()
fig.savefig('images/participant_demographics_chart.png', dpi=300)
plt.close(fig)

all_metrics = {**usability, **relevance, **explainability}
fig, ax = plt.subplots(figsize=(12, 7))
labels = list(all_metrics.keys())
values = list(all_metrics.values())
ax.plot(labels, values, marker='o', linestyle='-', color='#0057b7')
ax.fill_between(labels, values, color='#0057b7', alpha=0.15)
ax.set_title('Evaluation Profile Across All Metrics', fontsize=14, pad=12)
ax.set_xlabel('Evaluation metric')
ax.set_ylabel('Mean score (1–5)')
ax.set_ylim(0, 5)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.grid(axis='y', linestyle='--', alpha=0.4)
fig.tight_layout()
fig.savefig('images/evaluation_profile_chart.png', dpi=300)
plt.close(fig)

print('Saved 6 chart images to images/')
