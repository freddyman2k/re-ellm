from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

class TextEmbedder:
    """Uses a pretrained SBERT model to embed text into a vector representation that can be used by a SB3 Policy network. 
    Caches embeddings for each input to avoid recomputing them.
    """
    def __init__(self, model_name='paraphrase-MiniLM-L3-v2'):
        self.model = SentenceTransformer(model_name)
        self.embeddings = {}
    
    def embed(self, text):
        if text in self.embeddings:
            return self.embeddings[text]
        else:
            embedding = self.model.encode(text)
            self.embeddings[text] = embedding
            return embedding
        
def visualize_obs(obs, goal_suggestions=None, frame_stack=4):
    fig, axs = plt.subplots(1, frame_stack, figsize=(5 * frame_stack, 5))
    title = obs["text_obs"] if isinstance(obs["text_obs"], str) else f"text_obs is embedded: {obs['text_obs'].shape}" 
    if goal_suggestions is not None:
        goal_suggestions = '\n'.join(goal_suggestions)
        title += f"\nGoal suggestions: {goal_suggestions}"
    fig.suptitle(title , fontsize=16,weight="bold" )
    for i, frame in enumerate(obs["obs"][:frame_stack]):
        axs[i].imshow(frame)
        axs[i].axis('off')
        axs[i].set_title(f't={i - frame_stack + 1}', y=-0.15)
    plt.show()
