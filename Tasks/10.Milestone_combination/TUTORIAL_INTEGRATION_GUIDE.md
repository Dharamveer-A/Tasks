# 📺 Tutorial GIF/Video Integration Guide

## 🎯 Where to Add Tutorial GIF

The tutorial section has been added **right after the main header** and **before the file upload section** (around line 50-80 in the code).

Look for this section in the code:
```python
# ---------------- TUTORIAL VIDEO SECTION ----------------
st.markdown("---")

# Create an expander for the tutorial
with st.expander("📺 How to Use - Watch Tutorial", expanded=False):
```

---

## 🎬 Four Options to Add Your Tutorial

### **Option 1: Local GIF File (Recommended)**

If you have a GIF file saved locally:

```python
with st.expander("📺 How to Use - Watch Tutorial", expanded=False):
    st.markdown("""
    ### 🎬 Quick Tutorial
    Watch this short video to learn how to use the Skill Gap Analysis Dashboard:
    """)
    
    # Add your GIF here
    st.image("tutorial.gif", caption="Step-by-step tutorial", use_container_width=True)
```

**File Structure:**
```
your_project_folder/
├── skill_gap_analysis_final.py
├── tutorial.gif                 # Place your GIF here
└── requirements.txt
```

---

### **Option 2: Local Video File (MP4/WebM)**

If you have a video file:

```python
with st.expander("📺 How to Use - Watch Tutorial", expanded=False):
    st.markdown("""
    ### 🎬 Quick Tutorial
    Watch this short video to learn how to use the Skill Gap Analysis Dashboard:
    """)
    
    # Add your video here
    video_file = open('tutorial.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)
```

**Supported Formats:** MP4, WebM, OGG

---

### **Option 3: YouTube Video**

If your tutorial is on YouTube:

```python
with st.expander("📺 How to Use - Watch Tutorial", expanded=False):
    st.markdown("""
    ### 🎬 Quick Tutorial
    Watch this short video to learn how to use the Skill Gap Analysis Dashboard:
    """)
    
    # Add YouTube URL
    st.video("https://www.youtube.com/watch?v=YOUR_VIDEO_ID")
```

---

### **Option 4: GIF from URL**

If your GIF is hosted online:

```python
with st.expander("📺 How to Use - Watch Tutorial", expanded=False):
    st.markdown("""
    ### 🎬 Quick Tutorial
    Watch this short video to learn how to use the Skill Gap Analysis Dashboard:
    """)
    
    # Add GIF URL
    st.image("https://your-domain.com/tutorial.gif", 
             caption="Step-by-step tutorial", 
             use_container_width=True)
```

---

## 🔧 Step-by-Step Implementation

### Step 1: Choose Your Option
Decide which option works best for you (local file, YouTube, or URL).

### Step 2: Locate the Tutorial Section
Open `skill_gap_analysis_final.py` and find this section:
```python
# ---------------- TUTORIAL VIDEO SECTION ----------------
```

### Step 3: Uncomment Your Chosen Option
Find the option you want and uncomment it.

### Step 4: Remove Placeholder Instructions
Once you add your GIF/video, you can remove the placeholder section.

### Step 5: Test Your Changes
Run the app and click on the "📺 How to Use - Watch Tutorial" expander!

---

## 📝 Complete Example with GIF

```python
# ---------------- TUTORIAL VIDEO SECTION ----------------
st.markdown("---")

with st.expander("📺 How to Use - Watch Tutorial", expanded=False):
    st.markdown("""
    ### 🎬 Quick Tutorial
    Watch this short video to learn how to use the Skill Gap Analysis Dashboard:
    """)
    
    st.image("tutorial.gif", 
             caption="Step-by-step tutorial - How to analyze your resume", 
             use_container_width=True)

st.markdown("---")
```

---

## 🎥 Creating Your Tutorial GIF

### Recommended Tools:
1. **Screen Recording:**
   - Windows: OBS Studio, ShareX
   - Mac: QuickTime Player, ScreenFlow
   - Online: Loom, Screencastify

2. **GIF Conversion:**
   - ezgif.com (free online converter)
   - GIPHY (create and host GIFs)

### Best Practices:
- ✅ Keep GIF under 10MB for fast loading
- ✅ Resolution: 1280x720 or 1920x1080
- ✅ Duration: 30-60 seconds max
- ✅ Add captions/annotations for clarity

---

Need help? The tutorial section is fully commented in the code with all options ready to use!
