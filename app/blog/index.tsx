import { useRouter } from 'expo-router';
import React, { useEffect, useState } from 'react';
import { ActivityIndicator, Dimensions, ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

type BlogListItem = {
  slug: string;
  title: string;
  summary?: string;
};

export default function BlogsList() {
  const safeAreaInsets = useSafeAreaInsets();
  const router = useRouter();
  const screenHeight = Dimensions.get("window").height;

  const [blogs, setBlogs] = useState<BlogListItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchBlogs = async () => {
      try {
        const res = await fetch('http://192.168.0.103:5000/api/blogs');
        if (!res.ok) throw new Error('Failed to fetch blogs');
        const data = await res.json();
        setBlogs(data);
      } catch (err) {
        console.error(err);
        setBlogs([]);
      } finally {
        setLoading(false);
      }
    };
    fetchBlogs();
  }, []);

  if (loading) {
    return (
      <View style={[styles.container, styles.centered]}>
        <ActivityIndicator size="large" color="#4b7bec" />
        <Text>Loading...</Text>
      </View>
    );
  }

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={{
        paddingTop: safeAreaInsets.top + screenHeight * 0.06,
        paddingBottom: safeAreaInsets.bottom + screenHeight * 0.04,
        paddingHorizontal: 20
      }}
    >
      <Text style={styles.header}>Skin Health Blogs</Text>

      {blogs.length === 0 ? (
        <Text>No blogs found.</Text>
      ) : (
        blogs.map((blog) => (
          <TouchableOpacity
            key={blog.slug}
            style={styles.blogCard}
            onPress={() => router.push(`/blog/${blog.slug}`)}
          >
            <Text style={styles.title}>{blog.title}</Text>
            {blog.summary ? <Text style={styles.summary}>{blog.summary}</Text> : null}
          </TouchableOpacity>
        ))
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F5FCFF' },
  centered: { justifyContent: 'center', alignItems: 'center' },
  header: { fontSize: 28, fontWeight: 'bold', color: '#4b7bec', marginBottom: 20 },
  blogCard: { backgroundColor: '#fff', padding: 15, borderRadius: 15, marginBottom: 15, elevation: 3 },
  title: { fontSize: 20, fontWeight: 'bold', color: '#2d3436', marginBottom: 5 },
  summary: { fontSize: 16, color: '#778ca3' }
});
