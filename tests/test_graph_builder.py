import torch

from src.utils.graph_builder import DEMO_DIM, NOTE_EMB_DIM, build_patient_graph


def test_build_patient_graph_orders_nodes_and_only_connects_forward_in_time() -> None:
    graph = build_patient_graph(
        stay_id=7,
        note_rows=[{"note_id": "n1", "note_time": 2.0}],
        signal_rows=[
            {"item_type_id": 3, "norm_value": 0.5, "event_hours_from_intime": 1.0},
            {"item_type_id": 4, "norm_value": -0.5, "event_hours_from_intime": 1.0},
        ],
        note_embeddings={"n1": torch.ones(NOTE_EMB_DIM)},
        demo_feat=torch.zeros(DEMO_DIM),
        icd_feat=torch.ones(19),
    )

    assert graph is not None
    assert graph.node_type.tolist() == [2, 0, 0, 1]
    assert graph.x.shape == (4, NOTE_EMB_DIM)
    assert graph.demo.shape == (1, DEMO_DIM)

    edges = set(map(tuple, graph.edge_index.T.tolist()))
    assert (1, 2) not in edges
    assert edges == {(0, 1), (0, 2), (0, 3), (1, 3), (2, 3)}
    assert torch.all(graph.edge_attr > 0)


def test_build_patient_graph_returns_none_without_valid_nodes() -> None:
    graph = build_patient_graph(
        stay_id=7,
        note_rows=[{"note_id": "missing", "note_time": 2.0}],
        signal_rows=[],
        note_embeddings={},
    )

    assert graph is None
