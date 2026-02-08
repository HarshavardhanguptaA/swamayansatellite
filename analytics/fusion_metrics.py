def fusion_readiness(correlation, stability_x, stability_y):

    redundancy = abs(correlation)
    complementarity = 1 - redundancy
    stability_score = 1 / (stability_x["cv"] + stability_y["cv"])

    return {
        "redundancy_score": redundancy,
        "complementarity_score": complementarity,
        "fusion_stability": stability_score
    }
