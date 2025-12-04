import React, { useEffect, useState } from 'react';
import {
  Alert,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View
} from 'react-native';
import { useProfile } from './context/ProfileContext';

const ProfileScreen = () => {
  const { profile, updateProfile, updateSkinTone, loadProfile } = useProfile();
  const [isEditing, setIsEditing] = useState(false);
  const [tempProfile, setTempProfile] = useState(profile);

  useEffect(() => {
    setTempProfile(profile);
  }, [profile]);

  useEffect(() => {
    if (profile.username) {
      loadProfile(profile.username);
    }
  }, [profile.username]);

  const getFitzpatrickColor = (level: number): string => {
    const colors = {
      1: '#FFE4CC',
      2: '#F5D5B9', 
      3: '#E3B98B',
      4: '#C78A5E',
      5: '#8B4513',
      6: '#5D4037',
    };
    return colors[level as keyof typeof colors];
  };

  const getFitzpatrickDescription = (level: number): string => {
    const descriptions = {
      1: 'Type I: Pale white skin - Always burns, never tans',
      2: 'Type II: White skin - Burns easily, tans minimally',
      3: 'Type III: Light brown skin - Burns moderately, tans gradually',
      4: 'Type IV: Moderate brown skin - Burns minimally, tans well',
      5: 'Type V: Dark brown skin - Rarely burns, tans easily',
      6: 'Type VI: Deeply pigmented dark skin - Never burns',
    };
    return descriptions[level as keyof typeof descriptions];
  };

  const handleSkinToneUpdate = async (level: number) => {
    try {
      await updateSkinTone(level);
      setTempProfile({ ...tempProfile, fitzpatrickLevel: level });
      Alert.alert('Success', `Skin tone updated to Type ${level}`);
    } catch (error) {
      Alert.alert('Error', 'Failed to update skin tone. Please check your connection.');
    }
  };

  const handleSave = async () => {
    try {
      await updateProfile(tempProfile);
      Alert.alert('Success', 'Profile updated successfully');
      setIsEditing(false);
    } catch (error) {
      Alert.alert('Error', 'Failed to update profile. Please check your connection.');
    }
  };

  const handleCancel = () => {
    setTempProfile(profile);
    setIsEditing(false);
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Profile</Text>
        {!isEditing ? (
          <TouchableOpacity style={styles.editButton} onPress={() => setIsEditing(true)}>
            <Text style={styles.editButtonText}>Edit</Text>
          </TouchableOpacity>
        ) : (
          <View style={styles.editActions}>
            <TouchableOpacity style={styles.cancelButton} onPress={handleCancel}>
              <Text style={styles.cancelButtonText}>Cancel</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.saveButton} onPress={handleSave}>
              <Text style={styles.saveButtonText}>Save</Text>
            </TouchableOpacity>
          </View>
        )}
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Personal Information</Text>
        
        <View style={styles.inputGroup}>
          <Text style={styles.label}>Username</Text>
          <View style={[styles.input, styles.disabledInput]}>
            <Text style={styles.disabledText}>{profile.username}</Text>
          </View>
        </View>

        {/* <View style={styles.inputGroup}>
          <Text style={styles.label}>Email</Text>
          <View style={[styles.input, styles.disabledInput]}>
            <Text style={styles.disabledText}>{profile.email}</Text>
          </View>
        </View> */}

        {/* ADD AGE FIELD */}
        <View style={styles.inputGroup}>
          <Text style={styles.label}>Age</Text>
          {isEditing ? (
            <TextInput
              style={styles.input}
              value={tempProfile.age}
              onChangeText={(text) => setTempProfile({ ...tempProfile, age: text })}
              placeholder="Enter your age"
              keyboardType="numeric"
              maxLength={3}
            />
          ) : (
            <View style={[styles.input, styles.disabledInput]}>
              <Text style={styles.disabledText}>
                {profile.age ? `${profile.age} years` : 'Not specified'}
              </Text>
            </View>
          )}
        </View>

        <View style={styles.inputGroup}>
          <Text style={styles.label}>Gender</Text>
          {isEditing ? (
            <View style={styles.genderContainer}>
              {['Male', 'Female'].map((genderOption) => (
                <TouchableOpacity
                  key={genderOption}
                  style={[
                    styles.genderOption,
                    tempProfile.gender === genderOption && styles.genderOptionSelected
                  ]}
                  onPress={() => setTempProfile({ ...tempProfile, gender: genderOption })}
                >
                  <Text style={[
                    styles.genderOptionText,
                    tempProfile.gender === genderOption && styles.genderOptionTextSelected
                  ]}>
                    {genderOption}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          ) : (
            <View style={[styles.input, styles.disabledInput]}>
              <Text style={styles.disabledText}>
                {profile.gender || 'Not specified'}
              </Text>
            </View>
          )}
        </View>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Skin Profile</Text>
        
        <View style={styles.skinToneSection}>
          <Text style={styles.label}>Fitzpatrick Skin Tone Scale</Text>
          
          <View style={styles.scaleContainer}>
            {[1, 2, 3, 4, 5, 6].map((level) => (
              <TouchableOpacity
                key={level}
                style={[
                  styles.scaleItem,
                  { 
                    backgroundColor: getFitzpatrickColor(level),
                    borderColor: tempProfile.fitzpatrickLevel === level ? '#007AFF' : '#E5E5EA',
                    borderWidth: tempProfile.fitzpatrickLevel === level ? 3 : 1,
                  }
                ]}
                onPress={() => handleSkinToneUpdate(level)}
              >
                <Text style={[
                  styles.scaleText,
                  { color: level <= 2 ? '#000' : '#FFF' }
                ]}>
                  {level}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
          
          <Text style={styles.scaleDescription}>
            {getFitzpatrickDescription(tempProfile.fitzpatrickLevel)}
          </Text>
        </View>

        <View style={styles.inputGroup}>
          <Text style={styles.label}>Skin Type</Text>
          <TextInput
            style={[styles.input, !isEditing && styles.disabledInput]}
            value={tempProfile.skinType}
            onChangeText={(text) => setTempProfile({ ...tempProfile, skinType: text })}
            editable={isEditing}
            placeholder="e.g., Normal, Sensitive, Oily, Dry, Combination, Scaly"
          />
        </View>

        <View style={styles.inputGroup}>
          <Text style={styles.label}>Skin Conditions</Text>
          <TextInput
            style={[styles.input, !isEditing && styles.disabledInput]}
            value={tempProfile.skinConditions.join(', ')}
            onChangeText={(text) => setTempProfile({ 
              ...tempProfile, 
              skinConditions: text.split(',').map(item => item.trim()) 
            })}
            editable={isEditing}
            placeholder="e.g., Acne,  Hyperpigmentation, Large Spots, Skin Moles"
          />
        </View>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    padding: 16,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 24,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#1a1a1a',
  },
  editButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 8,
  },
  editButtonText: {
    color: '#fff',
    fontWeight: '600',
  },
  editActions: {
    flexDirection: 'row',
    gap: 12,
  },
  cancelButton: {
    backgroundColor: '#8E8E93',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 8,
  },
  cancelButtonText: {
    color: '#fff',
    fontWeight: '600',
  },
  saveButton: {
    backgroundColor: '#34C759',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 8,
  },
  saveButtonText: {
    color: '#fff',
    fontWeight: '600',
  },
  section: {
    marginBottom: 32,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 16,
    color: '#1a1a1a',
  },
  inputGroup: {
    marginBottom: 16,
  },
  label: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 8,
    color: '#333',
  },
  input: {
    borderWidth: 1,
    borderColor: '#E5E5EA',
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    backgroundColor: '#fff',
  },
  disabledInput: {
    backgroundColor: '#f5f5f5',
  },
  disabledText: {
    color: '#666',
    fontSize: 16,
  },
  genderContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  genderOption: {
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: '#E5E5EA',
    backgroundColor: '#fff',
  },
  genderOptionSelected: {
    backgroundColor: '#007AFF',
    borderColor: '#007AFF',
  },
  genderOptionText: {
    fontSize: 14,
    color: '#333',
    fontWeight: '500',
  },
  genderOptionTextSelected: {
    color: '#fff',
  },
  skinToneSection: {
    marginBottom: 24,
  },
  scaleContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    marginVertical: 12,
  },
  scaleItem: {
    width: '30%',
    height: 60,
    borderRadius: 8,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.2,
    shadowRadius: 1,
    elevation: 2,
  },
  scaleText: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  scaleDescription: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    marginTop: 8,
    fontStyle: 'italic',
  },
});

export default ProfileScreen;
