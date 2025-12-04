import { useLocalSearchParams } from 'expo-router';
import React, { useEffect, useState } from 'react';
import { ActivityIndicator, Image, Linking, ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';

type BlogContentItem = {
  letter?: string;
  name?: string;
  description?: string;
};

type BlogReference = {
  url: string;
  label: string;
};

type Blog = {
  title: string;
  summary?: string;
  image_url?: string;
  content?: BlogContentItem[];
  references?: BlogReference[]; // note plural
};

export default function BlogDetail() {
  const { slug } = useLocalSearchParams<{ slug: string }>();
  const [blog, setBlog] = useState<Blog | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!slug) return;
    const fetchBlog = async () => {
      try {
        const res = await fetch(`http://192.168.0.103:5000/api/blogs/${slug}`); // use HTTP for local dev
        if (!res.ok) throw new Error('Failed to fetch blog');
        const data = await res.json();
        setBlog(data);
      } catch (err) {
        console.error(err);
        setBlog(null);
      } finally {
        setLoading(false);
      }
    };
    fetchBlog();
  }, [slug]);

  if (loading) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color="#4b7bec" />
        <Text>Loading...</Text>
      </View>
    );
  }

  if (!blog) {
    return (
      <View style={styles.centered}>
        <Text>Blog not found.</Text>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={{ padding: 20 }}>
      <Text style={styles.title}>{blog.title}</Text>
      {blog.summary ? <Text style={styles.summary}>{blog.summary}</Text> : null}

      {blog.image_url && <Image source={{ uri: blog.image_url }} style={styles.image} />}

      {Array.isArray(blog.content) && blog.content.length > 0 ? (
        blog.content.map((item, idx) => (
          <View key={idx} style={styles.contentBlock}>
            {item.letter && <Text style={styles.letter}>{item.letter}</Text>}
            {item.name && <Text style={styles.name}>{item.name}</Text>}
            {item.description && <Text style={styles.description}>{item.description}</Text>}
          </View>
        ))
      ) : (
        <Text>No content available.</Text>
      )}

      {/* Render multiple references safely */}
      {Array.isArray(blog.references) && blog.references.length > 0 &&
        blog.references.map((ref, idx) =>
          ref.url && ref.label ? (
            <TouchableOpacity key={idx} onPress={() => Linking.openURL(ref.url)}>
              <Text style={styles.reference}>{ref.label}</Text>
            </TouchableOpacity>
          ) : null
        )
      }
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F5FCFF' },
  centered: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#F5FCFF' },
  title: { fontSize: 28, fontWeight: 'bold', color: '#4b7bec', marginBottom: 10 },
  summary: { fontSize: 16, color: '#778ca3', marginBottom: 15 },
  image: { width: '100%', height: 200, resizeMode: 'contain', marginBottom: 15 },
  contentBlock: { marginBottom: 12 },
  letter: { fontSize: 20, fontWeight: 'bold', color: '#2d3436' },
  name: { fontSize: 18, fontWeight: '600', color: '#2d3436' },
  description: { fontSize: 16, color: '#2d3436' },
  reference: { fontSize: 16, color: '#4b7bec', textDecorationLine: 'underline', marginTop: 10 }
});
