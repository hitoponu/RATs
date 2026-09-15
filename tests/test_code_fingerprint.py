"""Code fingerprints: does an attempt look like the one before it?"""

from __future__ import annotations

from rats.step_growth.code_fingerprint import (
    ast_hash,
    detect_collapse,
    fingerprint,
    summarize,
)

AVAIL = ["get_observation", "segment_sam3_text_prompt", "plan_grasp", "goto_pose",
         "close_gripper", "open_gripper", "get_oriented_bounding_box_from_3d_points",
         "point_prompt_molmo"]

# The run1 shape: mask centroid + hard-coded identity quaternion, no planner.
HANDBUILT = '''def main():
    with step_context("step-1", "grasp", step_index=0):
        obs = get_observation()
        masks = segment_sam3_text_prompt(obs["agentview"]["images"]["rgb"], "black bowl")
        centroid = np.array([0.1, 0.2, 0.95])
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        goto_pose(centroid, quat, z_approach=0.1)
        close_gripper()
'''

# Same behaviour, different variable names -- structurally identical.
HANDBUILT_RENAMED = '''def main():
    with step_context("step-1", "grasp", step_index=0):
        observation = get_observation()
        found = segment_sam3_text_prompt(observation["agentview"]["images"]["rgb"], "black bowl")
        target_xyz = np.array([0.1, 0.2, 0.95])
        orientation = np.array([1.0, 0.0, 0.0, 0.0])
        goto_pose(target_xyz, orientation, z_approach=0.1)
        close_gripper()
'''

PLANNED = '''def main():
    with step_context("step-1", "grasp", step_index=0):
        obs = get_observation()
        masks = segment_sam3_text_prompt(obs["agentview"]["images"]["rgb"], "black bowl")
        grasps, scores = plan_grasp(obs["agentview"]["images"]["depth"],
                                    obs["agentview"]["intrinsics"], masks[0]["mask"])
        world = obs["agentview"]["pose_mat"] @ grasps[int(np.argmax(scores))]
        goto_pose(world[:3, 3], mat_to_quat(world[:3, :3]), z_approach=0.1)
        close_gripper()
'''

OBB = '''def main():
    obs = get_observation()
    masks = segment_sam3_text_prompt(obs["agentview"]["images"]["rgb"], "black bowl")
    obb = get_oriented_bounding_box_from_3d_points(points)
    goto_pose(obb["center"], yaw_to_quat(obb["R"]))
    close_gripper()
'''


def fp(code: str) -> dict:
    return fingerprint(code, AVAIL, learned_skill_names=["grasp_helper"])


def test_identity_quaternion_and_planner_use_are_detected():
    hand, planned = fp(HANDBUILT), fp(PLANNED)
    assert hand["identity_quat"] is True and hand["uses_plan_grasp"] is False
    assert planned["identity_quat"] is False and planned["uses_plan_grasp"] is True
    assert hand["family"] == "handbuilt_identity" and planned["family"] == "graspnet"
    assert fp(OBB)["family"] == "obb_yaw"
    assert hand["ast_hash"] and hand["ast_hash"] != planned["ast_hash"]


def test_identity_quaternion_spellings():
    for literal in ("[1, 0, 0, 0]", "[1.0,0.0,0.0,0.0]", "(0, 0, 0, 1)", "[ 0.0 , 0.0 , 0.0 , 1.0 ]"):
        assert fingerprint(f"q = np.array({literal})", AVAIL)["identity_quat"], literal
    for literal in ("[1, 0, 0]", "[0.7, 0.0, 0.7, 0.0]", "[1, 0, 0, 0, 0]"):
        assert not fingerprint(f"q = np.array({literal})", AVAIL)["identity_quat"], literal


def test_renaming_variables_keeps_the_structure_hash():
    assert fp(HANDBUILT)["ast_hash"] == fp(HANDBUILT_RENAMED)["ast_hash"]
    # ... but swapping the pipeline does not
    assert fp(HANDBUILT)["ast_hash"] != fp(OBB)["ast_hash"]


def test_api_and_skill_calls_are_counted_separately():
    code = "grasp_helper('bowl')\nclose_gripper()\nclose_gripper()\nverified_close_gripper()\n"
    out = fingerprint(code, AVAIL, learned_skill_names=["grasp_helper"])
    assert out["api_calls"] == {"close_gripper": 2}       # not the look-alike name
    assert out["skill_calls"] == {"grasp_helper": 1}


def test_markers_are_counted_and_syntax_errors_are_survivable():
    assert fp(HANDBUILT)["step_markers"] == 1
    broken = fingerprint("def main(:\n  pass", AVAIL)
    assert broken["ast_hash"] == "" and broken["family"] == "other"


def test_summarize_reports_collapse_shaped_numbers():
    fps = [fp(HANDBUILT)] * 4 + [fp(PLANNED)]
    s = summarize(fps)
    assert s["n"] == 5
    assert s["identity_quat_frac"] == 0.8 and s["plan_grasp_frac"] == 0.2
    assert s["unique_ast_ratio"] == 0.4          # 2 distinct shapes over 5 attempts
    assert s["families"] == {"graspnet": 1, "handbuilt_identity": 4}
    assert 0.0 < s["family_entropy"] < 1.0
    assert s["marker_coverage"] == 1.0
    assert summarize([])["n"] == 0


def test_detect_collapse_fires_at_k_and_not_before():
    hand = fp(HANDBUILT)
    assert detect_collapse([hand] * 4, k=5) is None
    assert detect_collapse([hand] * 5, k=5) == "handbuilt_identity"
    assert detect_collapse([hand] * 4 + [fp(PLANNED)], k=5) is None
    # one family, but perception-derived orientations: not a collapse
    assert detect_collapse([fp(PLANNED)] * 5, k=5) is None
    assert detect_collapse([hand] * 5, k=0) is None


def test_ast_hash_keeps_known_function_names():
    a = ast_hash("plan_grasp(x)", keep={"plan_grasp"})
    b = ast_hash("other_call(x)", keep={"plan_grasp"})
    assert a and a != b
    # unknown names normalise to the same shape
    assert ast_hash("foo(x)") == ast_hash("bar(y)")
