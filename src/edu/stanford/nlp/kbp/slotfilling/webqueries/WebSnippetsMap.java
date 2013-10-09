package edu.stanford.nlp.kbp.slotfilling.webqueries;

import edu.stanford.nlp.util.CollectionFactory;
import edu.stanford.nlp.util.CollectionValuedMap;

import java.util.*;

/**
 * Collection of web snippets
 *
 * @author Angel Chang
 */
public class WebSnippetsMap {
  CollectionValuedMap<String, RelationMentionSnippets> slotSnippetsMap;
  CollectionValuedMap<String, RelationMentionSnippets> mentionSnippetsMap;

  public WebSnippetsMap()
  {
    slotSnippetsMap = new CollectionValuedMap<String, RelationMentionSnippets>(
            CollectionFactory.<RelationMentionSnippets>arrayListFactory());
    mentionSnippetsMap = new CollectionValuedMap<String, RelationMentionSnippets>(
            CollectionFactory.<RelationMentionSnippets>arrayListFactory());
  }

  public Collection<RelationMentionSnippets> getSnippetsBySlot(String slot)
  {
    return slotSnippetsMap.get(slot);
  }

  public Collection<RelationMentionSnippets> getSnippetsByEntity(String entity)
  {
    return mentionSnippetsMap.get(entity);
  }

  public void addSnippets(RelationMentionSnippets snippets)
  {/*
    Collection<RelationMentionSnippets> relationSnippets = slotSnippetsMap.get(snippets.mention.slotName);
    if (relationSnippets == null) {
      relationSnippets = new ArrayList<RelationMentionSnippets>();
      slotSnippetsMap.put(snippets.mention.slotName, relationSnippets);
    }
    relationSnippets.add(snippets); */
    slotSnippetsMap.add(snippets.slotName, snippets);
    slotSnippetsMap.add(snippets.entityName, snippets);
  }

}
