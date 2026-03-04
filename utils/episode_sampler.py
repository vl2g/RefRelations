"""
utils/episode_sampler.py

Episodic sampler that builds K-shot episodes from a predicate–video index.
Can be used independently of the Dataset class for custom training loops.
"""
from __future__ import annotations
import random
from typing import Dict, List, Optional, Tuple


class EpisodeSampler:
    """
    Samples N-way K-shot episodes for meta-training / meta-testing.

    Args:
        predicate2videos : dict mapping predicate → list of video dicts
        support_size     : K, number of support videos per episode
        n_way            : (unused in the paper's 1-way setup; kept for extension)
    """

    def __init__(
        self,
        predicate2videos: Dict[str, List[dict]],
        support_size: int = 4,
        n_way: int = 1,
        seed: Optional[int] = None,
    ) -> None:
        self.predicate2videos = predicate2videos
        self.support_size     = support_size
        self.n_way            = n_way
        self.rng              = random.Random(seed)

        # Only keep predicates with enough videos
        self.valid_predicates = [
            p for p, vids in predicate2videos.items()
            if len(vids) >= support_size + 1
        ]

    def sample_episode(self, predicate: Optional[str] = None) -> dict:
        """
        Sample a single episode.

        Returns:
            {
                "predicate"     : str,
                "test_video"    : dict,
                "support_videos": List[dict]
            }
        """
        if predicate is None:
            predicate = self.rng.choice(self.valid_predicates)

        videos = self.predicate2videos[predicate]
        sampled = self.rng.sample(videos, min(self.support_size + 1, len(videos)))
        test_video     = sampled[0]
        support_videos = sampled[1:]

        return {
            "predicate":      predicate,
            "test_video":     test_video,
            "support_videos": support_videos,
        }

    def generate_episodes(self, n_episodes: int) -> List[dict]:
        """Generate a fixed list of episodes (for reproducible evaluation)."""
        return [self.sample_episode() for _ in range(n_episodes)]

    def __repr__(self) -> str:
        return (
            f"EpisodeSampler("
            f"n_predicates={len(self.valid_predicates)}, "
            f"K={self.support_size})"
        )
