from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

from studyforge.store import Manifest, vector_store


def test_manifest_roundtrip(db_settings):
    m = Manifest(db_settings)
    assert m.all() == {}
    m.put("UiO/A/x.md", "sha1", ["UiO/A/x.md"])
    m.put("UiO/A/x.md", "sha2", ["UiO/A/x.md#p1", "UiO/A/x.md#p2"])  # upsert
    assert m.all() == {"UiO/A/x.md": ("sha2", ["UiO/A/x.md#p1", "UiO/A/x.md#p2"])}
    m.remove("UiO/A/x.md")
    assert m.all() == {}


def test_vector_store_add_and_delete_by_ref_doc(db_settings):
    vs = vector_store(db_settings)
    node = TextNode(
        text="hello", embedding=[0.1] * 1024, metadata={"path": "p"},
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc-1")},  # ref_doc_id = doc-1
    )
    vs.add([node])
    assert len(vs.get_nodes(node_ids=[node.node_id])) == 1
    vs.delete("doc-1")
    assert vs.get_nodes(node_ids=[node.node_id]) == []
