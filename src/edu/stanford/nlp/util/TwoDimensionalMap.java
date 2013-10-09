package edu.stanford.nlp.util;

import java.util.*;
import java.io.Serializable;

import edu.stanford.nlp.util.MapFactory;

/**
 * @author grenager
 */
public class TwoDimensionalMap<K1, K2, V> implements Serializable {

  private static final long serialVersionUID = 1L;
  Map<K1, HashMap<K2, V>> map;

  public int size() {
    return map.size();
  }

  public V put(K1 key1, K2 key2, V value) {
    Map<K2, V> m = getMap(key1);
    return m.put(key2, value);
  }

  // adds empty hashmap for key key1
  public void put(K1 key1) {
    map.put(key1, new HashMap<K2, V>());
  }

  public V get(K1 key1, K2 key2) {
    Map<K2, V> m = getMap(key1);
    return m.get(key2);
  }

  public void remove(K1 key1, K2 key2) {
    get(key1).remove(key2);
  }

  // removes almost the associated data with the key in the first map
  public void remove(K1 key1) {
    map.remove(key1);
  }

  public void clear() {
    map.clear();
  }

  public boolean containsKey(K1 key1) {
    return map.containsKey(key1);
  }

  public Map<K2, V> get(K1 key1) {
    return getMap(key1);
  }

  public Map<K2, V> getMap(K1 key1) {
    HashMap<K2, V> m = map.get(key1);
    if (m == null) {
      m = new HashMap<K2, V>();
      map.put(key1, m);
    }
    return m;
  }

  public Collection<V> values() {
    Set<V> s = Generics.newHashSet();
    for (HashMap<K2, V> innerMap : map.values()) {
      s.addAll(innerMap.values());
    }
    return s;
  }

  public Set<K1> firstKeySet() {
    return map.keySet();
  }

  public Set<K2> secondKeySet() {
    Set<K2> keys = Generics.newHashSet();
    for (K1 k1 : map.keySet()) {
      keys.addAll(get(k1).keySet());
    }
    return keys;
  }

  public Set<Map.Entry<K1, HashMap<K2, V>>> entrySet() {
    return map.entrySet();
  }

  public TwoDimensionalMap() {
    this.map = new HashMap<K1, HashMap<K2, V>>();
  }

  public TwoDimensionalMap(TwoDimensionalMap<K1, K2, V> tdm) {
    this.map = new HashMap<K1, HashMap<K2, V>>();
    for (K1 k1 : tdm.map.keySet()) {
      HashMap<K2, V> m = tdm.map.get(k1);
      this.map.put(k1, new HashMap<K2, V>(m));
    }
  }

  @SuppressWarnings("unchecked")
  public TwoDimensionalMap(MapFactory mf) {
    this.map = Generics.newHashMap();
  }

  @Override
  public String toString() {
    return map.toString();
  }

}
