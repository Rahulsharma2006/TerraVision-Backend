
def classify_change(before_stats, after_stats, temporal_result):
    """
    Transparent rules layer. It is intentionally replaceable by a trained
    change-type classifier once labelled SIH data are available.
    """
    ratio=temporal_result.get("change_ratio",0)
    regions=temporal_result.get("regions",[])
    area=max([r["relative_area"] for r in regions],default=0)

    if ratio<0.005:
        label="no_change"
    elif area>0.20:
        label="large_area_change"
    elif len(regions)>=3:
        label="multiple_local_changes"
    else:
        label="localized_change"

    return {
        "label":label,
        "confidence":float(temporal_result.get("confidence",0)),
        "explainable_rules":[
            "change ratio",
            "connected-component count",
            "largest changed-region relative area"
        ]
    }
