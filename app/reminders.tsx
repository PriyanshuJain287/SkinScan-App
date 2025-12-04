import { Feather, MaterialCommunityIcons } from '@expo/vector-icons';
import DateTimePicker from '@react-native-community/datetimepicker';
import * as Notifications from 'expo-notifications';
import React, { useEffect, useState } from 'react';
import {
    Alert,
    Modal,
    Platform,
    ScrollView,
    StyleSheet,
    Switch,
    Text,
    TouchableOpacity,
    View,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

interface Reminder {
  id: string;
  enabled: boolean;
  time: Date;
  frequency: 'daily' | 'weekly' | 'monthly';
  title: string;
  description: string;
}

interface Tip {
  id: string;
  title: string;
  shortDescription: string;
  detailedDescription: string;
  icon: string;
}

export default function RemindersScreen() {
  const safeAreaInsets = useSafeAreaInsets();
  const [reminders, setReminders] = useState<Reminder[]>([]);
  const [showTimePicker, setShowTimePicker] = useState<string | null>(null);
  const [selectedTip, setSelectedTip] = useState<Tip | null>(null);
  const [modalVisible, setModalVisible] = useState(false);

  // Self-check tips data
  const selfCheckTips: Tip[] = [
    {
      id: '1',
      title: 'Proper Lighting & Mirror',
      shortDescription: 'Use good lighting and a full-length mirror',
      detailedDescription: 'Perform your skin check in a well-lit room with both a hand mirror and full-length mirror. Natural daylight is best. Use the hand mirror to examine hard-to-see areas like your back, scalp, and the backs of your thighs.',
      icon: 'sun',
    },
    {
      id: '2',
      title: 'Complete Body Examination',
      shortDescription: 'Check your entire body, including hard-to-see areas',
      detailedDescription: 'Systematically examine your entire body:\n\n• Face, ears, neck, and scalp\n• Front and back of torso\n• Arms, including underarms and palms\n• Legs, including between toes and soles\n• Back, buttocks, and genital area\n\nDon\'t forget areas not exposed to the sun!',
      icon: 'eye',
    },
    {
      id: '3',
      title: 'Monitor Changes',
      shortDescription: 'Look for new moles or changes in existing ones',
      detailedDescription: 'Regularly monitor your moles and spots for any changes. Keep track of:\n\n• New moles that appear\n• Changes in size, shape, or color of existing moles\n• Moles that itch, bleed, or become painful\n• Spots that look different from others\n\nTake photos monthly to compare changes over time.',
      icon: 'trending-up',
    },
    {
      id: '4',
      title: 'ABCDE Rule',
      shortDescription: 'Use the ABCDE rule for mole assessment',
      detailedDescription: 'Follow the ABCDE rule to identify potential warning signs:\n\n• Asymmetry: One half doesn\'t match the other\n• Border: Irregular, ragged, or blurred edges\n• Color: Uneven color or multiple shades\n• Diameter: Larger than 6mm (pencil eraser size)\n• Evolving: Changing in size, shape, or color\n\nIf you notice any of these signs, consult a dermatologist.',
      icon: 'alert-triangle',
    },
    {
      id: '5',
      title: 'Document Findings',
      shortDescription: 'Keep a record of your self-examinations',
      detailedDescription: 'Maintain a skin health journal:\n\n• Date each self-examination\n• Note any new or changing spots\n• Take photos for comparison\n• Record any concerns or questions\n• Track your sunscreen usage\n\nThis helps you and your doctor monitor changes over time.',
      icon: 'file-text',
    },
    {
      id: '6',
      title: 'Professional Check-ups',
      shortDescription: 'Schedule regular dermatologist visits',
      detailedDescription: 'In addition to self-exams:\n\n• See a dermatologist annually for a professional skin check\n• More frequently if you have risk factors\n• Family history of skin cancer\n• Many moles or atypical moles\n• History of sunburns\n• Fair skin that burns easily\n\nEarly detection saves lives!',
      icon: 'user-check',
    },
  ];

  // Configure notifications
  useEffect(() => {
    configureNotifications();
    loadReminders();
  }, []);

  const configureNotifications = async () => {
    try {
      const { status } = await Notifications.requestPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert(
          'Notification Permission',
          'Please enable notifications to receive reminder alerts.',
          [{ text: 'OK' }]
        );
      }

      Notifications.setNotificationHandler({
        handleNotification: async () => ({
          shouldShowAlert: true,
          shouldPlaySound: true,
          shouldSetBadge: false,
        }),
      });
    } catch (error) {
      console.log('Notification configuration error:', error);
    }
  };

  const loadReminders = () => {
    const defaultReminders: Reminder[] = [
      {
        id: '1',
        enabled: false,
        time: new Date(new Date().setHours(9, 0, 0, 0)),
        frequency: 'monthly',
        title: 'Monthly Full Body Check',
        description: 'Complete a thorough skin self-examination'
      },
      {
        id: '2',
        enabled: false,
        time: new Date(new Date().setHours(20, 0, 0, 0)),
        frequency: 'weekly',
        title: 'Weekly Spot Check',
        description: 'Quick check of any concerning spots'
      },
      {
        id: '3',
        enabled: false,
        time: new Date(new Date().setHours(19, 0, 0, 0)),
        frequency: 'daily',
        title: 'Daily Sunscreen Reminder',
        description: 'Remember to apply and reapply sunscreen'
      }
    ];
    setReminders(defaultReminders);
  };

  const scheduleNotification = async (reminder: Reminder) => {
    try {
      if (!reminder.enabled) return;

      let trigger: any = {
        hour: reminder.time.getHours(),
        minute: reminder.time.getMinutes(),
        repeats: true,
      };

      await Notifications.scheduleNotificationAsync({
        content: {
          title: reminder.title,
          body: reminder.description,
          sound: true,
          data: { reminderId: reminder.id },
        },
        trigger,
      });
      
      console.log(`Scheduled notification for ${reminder.title}`);
    } catch (error) {
      console.log('Error scheduling notification:', error);
      Alert.alert('Error', 'Failed to schedule notification');
    }
  };

  const cancelNotification = async (reminderId: string) => {
    try {
      await Notifications.cancelScheduledNotificationAsync(reminderId);
      console.log(`Cancelled notification for reminder ${reminderId}`);
    } catch (error) {
      console.log('Error cancelling notification:', error);
    }
  };

  const toggleReminder = async (reminderId: string) => {
    const updatedReminders = reminders.map(reminder => {
      if (reminder.id === reminderId) {
        const newEnabled = !reminder.enabled;
        
        if (newEnabled) {
          scheduleNotification({ ...reminder, enabled: true });
        } else {
          cancelNotification(reminderId);
        }
        
        return { ...reminder, enabled: newEnabled };
      }
      return reminder;
    });

    setReminders(updatedReminders);
    
    const reminder = updatedReminders.find(r => r.id === reminderId);
    if (reminder) {
      Alert.alert(
        'Reminder Updated', 
        `${reminder.title} is now ${reminder.enabled ? 'enabled' : 'disabled'}`,
        [{ text: 'OK' }]
      );
    }
  };

  const updateReminderTime = (reminderId: string, newTime: Date) => {
    const updatedReminders = reminders.map(reminder => {
      if (reminder.id === reminderId) {
        const updatedReminder = { ...reminder, time: newTime };
        
        if (reminder.enabled) {
          cancelNotification(reminderId);
          scheduleNotification(updatedReminder);
        }
        
        return updatedReminder;
      }
      return reminder;
    });

    setReminders(updatedReminders);
    setShowTimePicker(null);
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const getFrequencyText = (frequency: string) => {
    switch (frequency) {
      case 'daily': return 'Every day';
      case 'weekly': return 'Every week';
      case 'monthly': return 'Every month';
      default: return frequency;
    }
  };

  const getReminderIcon = (frequency: string) => {
    switch (frequency) {
      case 'daily': return 'calendar-today';
      case 'weekly': return 'calendar-week';
      case 'monthly': return 'calendar-month';
      default: return 'calendar';
    }
  };

  const handleTimeChange = (event: any, selectedTime?: Date) => {
    if (showTimePicker && selectedTime) {
      updateReminderTime(showTimePicker, selectedTime);
    }
    setShowTimePicker(null);
  };

  const openTipDetail = (tip: Tip) => {
    setSelectedTip(tip);
    setModalVisible(true);
  };

  const closeTipDetail = () => {
    setModalVisible(false);
    setSelectedTip(null);
  };

  const renderTipItem = (tip: Tip) => (
    <TouchableOpacity
      key={tip.id}
      style={styles.tipItem}
      onPress={() => openTipDetail(tip)}
      activeOpacity={0.7}
    >
      <View style={styles.tipIconContainer}>
        <Feather name={tip.icon as any} size={18} color="#4b7bec" />
      </View>
      <View style={styles.tipContent}>
        <Text style={styles.tipText}>{tip.shortDescription}</Text>
        <Feather name="chevron-right" size={16} color="#999" />
      </View>
    </TouchableOpacity>
  );

  return (
    <View style={[styles.container, { paddingTop: safeAreaInsets.top }]}>
      <View style={styles.header}>
        <Text style={styles.title}>Self-Check Reminders</Text>
        <Text style={styles.subtitle}>
          Set reminders for regular skin self-examinations
        </Text>
      </View>

      <ScrollView style={styles.remindersList} showsVerticalScrollIndicator={false}>
        {reminders.map((reminder) => (
          <View key={reminder.id} style={styles.reminderCard}>
            <View style={styles.reminderHeader}>
              <View style={styles.reminderIcon}>
                <MaterialCommunityIcons 
                  name={getReminderIcon(reminder.frequency)} 
                  size={24} 
                  color="#4b7bec" 
                />
              </View>
              <View style={styles.reminderInfo}>
                <Text style={styles.reminderTitle}>{reminder.title}</Text>
                <Text style={styles.reminderDescription}>{reminder.description}</Text>
                <View style={styles.reminderDetails}>
                  <Feather name="clock" size={14} color="#666" />
                  <Text style={styles.reminderDetailText}>
                    {formatTime(reminder.time)} • {getFrequencyText(reminder.frequency)}
                  </Text>
                </View>
              </View>
              <Switch
                value={reminder.enabled}
                onValueChange={() => toggleReminder(reminder.id)}
                trackColor={{ false: '#f0f0f0', true: '#4b7bec' }}
                thumbColor={reminder.enabled ? '#fff' : '#f4f3f4'}
              />
            </View>

            <View style={styles.reminderActions}>
              <TouchableOpacity
                style={styles.timeButton}
                onPress={() => setShowTimePicker(reminder.id)}
              >
                <Feather name="edit-2" size={16} color="#4b7bec" />
                <Text style={styles.timeButtonText}>Change Time</Text>
              </TouchableOpacity>
            </View>

            {showTimePicker === reminder.id && (
              <DateTimePicker
                value={reminder.time}
                mode="time"
                display={Platform.OS === 'ios' ? 'spinner' : 'default'}
                onChange={handleTimeChange}
                style={styles.timePicker}
              />
            )}
          </View>
        ))}

        <View style={styles.tipsContainer}>
          <View style={styles.tipsHeader}>
            <Text style={styles.tipsTitle}>Self-Check Guide</Text>
            <Feather name="info" size={20} color="#4b7bec" />
          </View>
          <Text style={styles.tipsSubtitle}>
            Tap on any tip to learn more about proper skin self-examination techniques
          </Text>
          {selfCheckTips.map(renderTipItem)}
        </View>
      </ScrollView>

      {/* Tip Detail Modal */}
      <Modal
        visible={modalVisible}
        animationType="slide"
        presentationStyle="pageSheet"
        onRequestClose={closeTipDetail}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <TouchableOpacity onPress={closeTipDetail} style={styles.closeButton}>
              <Feather name="x" size={24} color="#333" />
            </TouchableOpacity>
            <Text style={styles.modalTitle}>Self-Check Guide</Text>
            <View style={styles.closeButton} />
          </View>

          {selectedTip && (
            <ScrollView style={styles.modalContent} showsVerticalScrollIndicator={false}>
              <View style={styles.tipDetailHeader}>
                <View style={[styles.tipIconContainer, styles.tipDetailIcon]}>
                  <Feather name={selectedTip.icon as any} size={24} color="#4b7bec" />
                </View>
                <Text style={styles.tipDetailTitle}>{selectedTip.title}</Text>
              </View>
              
              <View style={styles.tipDetailDescription}>
                <Text style={styles.tipDetailText}>
                  {selectedTip.detailedDescription}
                </Text>
              </View>

              <View style={styles.tipActions}>
                <TouchableOpacity style={styles.gotItButton} onPress={closeTipDetail}>
                  <Text style={styles.gotItButtonText}>Got It</Text>
                </TouchableOpacity>
              </View>
            </ScrollView>
          )}
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
    paddingHorizontal: 16,
  },
  header: {
    alignItems: 'center',
    marginBottom: 20,
    paddingTop: 10,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
    marginTop: 8,
    lineHeight: 20,
  },
  remindersList: {
    flex: 1,
  },
  reminderCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  reminderHeader: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  reminderIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#e8eaf6',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
    marginTop: 2,
  },
  reminderInfo: {
    flex: 1,
    marginRight: 12,
  },
  reminderTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 4,
  },
  reminderDescription: {
    fontSize: 14,
    color: '#666',
    marginBottom: 8,
    lineHeight: 18,
  },
  reminderDetails: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  reminderDetailText: {
    fontSize: 12,
    color: '#666',
    marginLeft: 4,
  },
  reminderActions: {
    flexDirection: 'row',
    justifyContent: 'flex-start',
  },
  timeButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f8f9fa',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 20,
  },
  timeButtonText: {
    color: '#4b7bec',
    fontWeight: '600',
    marginLeft: 6,
    fontSize: 14,
  },
  timePicker: {
    marginTop: 12,
  },
  tipsContainer: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginTop: 20,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  tipsHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  tipsTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  tipsSubtitle: {
    fontSize: 14,
    color: '#666',
    marginBottom: 16,
    lineHeight: 18,
  },
  tipItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    paddingHorizontal: 8,
    borderRadius: 8,
    backgroundColor: '#f8f9fa',
    marginBottom: 8,
  },
  tipIconContainer: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: '#e8eaf6',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  tipContent: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  tipText: {
    fontSize: 14,
    color: '#333',
    fontWeight: '500',
    flex: 1,
  },
  // Modal Styles
  modalContainer: {
    flex: 1,
    backgroundColor: '#fff',
  },
  modalHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  closeButton: {
    padding: 4,
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  modalContent: {
    flex: 1,
    padding: 20,
  },
  tipDetailHeader: {
    alignItems: 'center',
    marginBottom: 24,
  },
  tipDetailIcon: {
    width: 60,
    height: 60,
    borderRadius: 30,
    marginBottom: 12,
  },
  tipDetailTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#333',
    textAlign: 'center',
  },
  tipDetailDescription: {
    backgroundColor: '#f8f9fa',
    borderRadius: 12,
    padding: 20,
    marginBottom: 24,
  },
  tipDetailText: {
    fontSize: 16,
    color: '#333',
    lineHeight: 24,
  },
  tipActions: {
    alignItems: 'center',
  },
  gotItButton: {
    backgroundColor: '#4b7bec',
    paddingHorizontal: 32,
    paddingVertical: 12,
    borderRadius: 25,
  },
  gotItButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});
