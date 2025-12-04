import { Ionicons, MaterialCommunityIcons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import React, { useEffect, useRef, useState } from 'react';
import {
    FlatList,
    KeyboardAvoidingView,
    Platform,
    StyleSheet,
    Text,
    TextInput,
    TouchableOpacity,
    View
} from 'react-native';

interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
}

const BACKEND_BASE_URL = 'http://192.168.0.103:5000';

export default function ChatbotScreen() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: "👋 Hello! I'm SkinAI, your personal dermatology assistant! \n\nI can help you with:\n• Skin analysis and understanding results\n• Finding nearby dermatologists\n• Skin health education and prevention\n• Setting up reminders\n• Navigating the app features\n\nI'm here to make your skin health journey easier! What would you like to know today? 🌟",
      isUser: false,
      timestamp: new Date(),
    },
  ]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const flatListRef = useRef<FlatList>(null);
  const router = useRouter();

  const quickActions = [
    { 
      icon: "camera", 
      text: "How to analyze skin", 
      prompt: "How do I use the skin analysis feature?" 
    },
    { 
      icon: "map-marker", 
      text: "Find dermatologists", 
      prompt: "How do I find nearby skin doctors?" 
    },
    { 
      icon: "book", 
      text: "Learn about skin cancer", 
      prompt: "Tell me about skin cancer prevention" 
    },
    { 
      icon: "bell", 
      text: "Set reminders", 
      prompt: "How do I set skin check reminders?" 
    },
  ];

  const sendMessage = async (customMessage?: string) => {
    const messageToSend = customMessage || inputText.trim();
    if (!messageToSend) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: messageToSend,
      isUser: true,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    if (!customMessage) setInputText('');
    setIsLoading(true);

    try {
      const response = await fetch(`${BACKEND_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: messageToSend,
          user_id: 'current-user',
        }),
      });

      const data = await response.json();

      if (data.success) {
        const aiMessage: Message = {
          id: (Date.now() + 1).toString(),
          text: data.response,
          isUser: false,
          timestamp: new Date(),
        };
        setMessages(prev => [...prev, aiMessage]);
      } else {
        throw new Error(data.error);
      }
    } catch (error) {
      console.error('Chat error:', error);
      
      // Enhanced error handling
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: "I'm here to help! 🌟 Let me tell you what I can do:\n\n• Guide you through skin analysis\n• Help find nearby dermatologists\n• Explain skin health topics\n• Assist with app navigation\n• Set up reminders\n\nWhat would you like help with today?",
        isUser: false,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

    const handleQuickAction = (prompt: string) => {
        sendMessage(prompt);
    };

    type ValidRoute = 'home' | 'hospitals' | 'blog';
    const navigateToScreen = (screen: ValidRoute) => {
    router.push(`/${screen}`);
    };

  useEffect(() => {
    if (messages.length > 0) {
      setTimeout(() => {
        flatListRef.current?.scrollToEnd({ animated: true });
      }, 100);
    }
  }, [messages]);

  const renderMessage = ({ item }: { item: Message }) => (
    <View style={[
      styles.messageContainer,
      item.isUser ? styles.userMessage : styles.aiMessage
    ]}>
      <View style={[
        styles.messageBubble,
        item.isUser ? styles.userBubble : styles.aiBubble
      ]}>
        <Text style={[
          styles.messageText,
          item.isUser ? styles.userMessageText : styles.aiMessageText
        ]}>
          {item.text}
        </Text>
        <Text style={styles.timestamp}>
          {item.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </Text>
      </View>
    </View>
  );

  return (
    <KeyboardAvoidingView 
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
    >
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.botInfo}>
          <View style={styles.botAvatar}>
            <Ionicons name="medical" size={24} color="#fff" />
          </View>
          <View>
            <Text style={styles.botName}>SkinAI Assistant</Text>
            <Text style={styles.botStatus}>
              {isLoading ? 'Thinking...' : 'Online • Ready to help!'}
            </Text>
          </View>
        </View>
        <TouchableOpacity 
          style={styles.backButton}
          onPress={() => router.back()}
        >
          <Ionicons name="arrow-back" size={24} color="#666" />
        </TouchableOpacity>
      </View>

      {/* Quick Actions */}
      {messages.length <= 1 && (
        <View style={styles.quickActionsContainer}>
          <Text style={styles.quickActionsTitle}>Quick Help</Text>
          <View style={styles.quickActionsGrid}>
            {quickActions.map((action, index) => (
              <TouchableOpacity
                key={index}
                style={styles.quickActionButton}
                onPress={() => handleQuickAction(action.prompt)}
              >
                <MaterialCommunityIcons name={action.icon as any} size={20} color="#4b7bec" />
                <Text style={styles.quickActionText}>{action.text}</Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>
      )}

      {/* Messages */}
      <FlatList
        ref={flatListRef}
        data={messages}
        renderItem={renderMessage}
        keyExtractor={item => item.id}
        style={styles.messagesList}
        contentContainerStyle={styles.messagesContent}
      />

      {/* Quick Navigation */}
      <View style={styles.navigationContainer}>
        <Text style={styles.navigationTitle}>Quick Navigation</Text>
        <View style={styles.navigationButtons}>
          <TouchableOpacity style={styles.navButton} onPress={() => navigateToScreen('home')}>
            <MaterialCommunityIcons name="camera" size={16} color="#fff" />
            <Text style={styles.navButtonText}>Scan Skin</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.navButton} onPress={() => navigateToScreen('hospitals')}>
            <MaterialCommunityIcons name="hospital" size={16} color="#fff" />
            <Text style={styles.navButtonText}>Hospitals</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.navButton} onPress={() => navigateToScreen('blog')}>
            <MaterialCommunityIcons name="book" size={16} color="#fff" />
            <Text style={styles.navButtonText}>Blog</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Input Area */}
      <View style={styles.inputContainer}>
        <TextInput
          style={styles.textInput}
          value={inputText}
          onChangeText={setInputText}
          placeholder="Ask about skin health or app features..."
          placeholderTextColor="#999"
          multiline
          maxLength={1000}
        />
        <TouchableOpacity
          style={[
            styles.sendButton,
            (!inputText.trim() || isLoading) && styles.sendButtonDisabled
          ]}
          onPress={() => sendMessage()}
          disabled={!inputText.trim() || isLoading}
        >
          <Ionicons 
            name={isLoading ? "time-outline" : "send"} 
            size={20} 
            color="#fff" 
          />
        </TouchableOpacity>
      </View>

      {/* Disclaimer */}
      <View style={styles.disclaimer}>
        <Text style={styles.disclaimerText}>
          ⚠️ AI assistant - Not medical advice. Always consult healthcare professionals for medical concerns.
        </Text>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  botInfo: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  botAvatar: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#4b7bec',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  botName: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
  },
  botStatus: {
    fontSize: 12,
    color: '#666',
  },
  backButton: {
    padding: 4,
  },
  quickActionsContainer: {
    padding: 16,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  quickActionsTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#666',
    marginBottom: 12,
  },
  quickActionsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  quickActionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f8f9fa',
    padding: 12,
    borderRadius: 8,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: '#e9ecef',
    flex: 1,
    marginHorizontal: 4,
    minWidth: '45%',
  },
  quickActionText: {
    fontSize: 12,
    color: '#495057',
    marginLeft: 6,
    fontWeight: '500',
  },
  navigationContainer: {
    padding: 16,
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
  },
  navigationTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#666',
    marginBottom: 8,
  },
  navigationButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  navButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#4b7bec',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 6,
    flex: 1,
    marginHorizontal: 4,
    justifyContent: 'center',
  },
  navButtonText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '500',
    marginLeft: 4,
  },
  messagesList: {
    flex: 1,
  },
  messagesContent: {
    padding: 16,
  },
  messageContainer: {
    marginBottom: 16,
    flexDirection: 'row',
  },
  userMessage: {
    justifyContent: 'flex-end',
  },
  aiMessage: {
    justifyContent: 'flex-start',
  },
  messageBubble: {
    maxWidth: '85%',
    padding: 12,
    borderRadius: 16,
  },
  userBubble: {
    backgroundColor: '#4b7bec',
    borderBottomRightRadius: 4,
  },
  aiBubble: {
    backgroundColor: '#fff',
    borderBottomLeftRadius: 4,
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  messageText: {
    fontSize: 16,
    lineHeight: 20,
  },
  userMessageText: {
    color: '#fff',
  },
  aiMessageText: {
    color: '#333',
  },
  timestamp: {
    fontSize: 11,
    color: '#999',
    marginTop: 4,
    alignSelf: 'flex-end',
  },
  inputContainer: {
    flexDirection: 'row',
    padding: 16,
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
    alignItems: 'flex-end',
  },
  textInput: {
    flex: 1,
    borderWidth: 1,
    borderColor: '#e0e0e0',
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 12,
    marginRight: 12,
    fontSize: 16,
    maxHeight: 100,
    backgroundColor: '#f8f9fa',
  },
  sendButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: '#4b7bec',
    justifyContent: 'center',
    alignItems: 'center',
  },
  sendButtonDisabled: {
    backgroundColor: '#ccc',
  },
  disclaimer: {
    padding: 12,
    backgroundColor: '#fff3cd',
    borderTopWidth: 1,
    borderTopColor: '#ffeaa7',
  },
  disclaimerText: {
    fontSize: 12,
    color: '#856404',
    textAlign: 'center',
    lineHeight: 16,
  },
});
