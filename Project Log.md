# Project Log

---

### 26.12.25

- Downloaded Pupil Core Bundle
- Got the glasses working (callibration, checking video feeds)
- Did more research on the Network API and IPC Backbone, and realised that the Pupil Remote needed to receive data from the glasses only runs along with Pupil Core Software. The Bela can’t run Pupil Service. Added Raspberry Pi to the planned stack to run between glasses and Bela. This will run Pupil Service and data processing (incl. potential ML models), only feeding numbers to the Bela over USB.
- Made development plan
    
    [Prototypes](https://www.notion.so/Prototypes-2d5693cc579c8046905fc3b812b209a7?pvs=21)
    

### 27.12.25

Time spent: ~2,5 h

- Set up initial repo using cursor on Jørgens mac
- Tested streaming from the glasses in own program with brightness tracker

```
Initial commit: Pupil Core brightness tracker

- Real-time gaze streaming from Pupil Capture via ZMQ
- Zero-buffer streaming for minimal latency (RCVHWM=1)
- Brightness analysis at gaze point with smoothing
- Extensible output interface (Console, File, Threshold sinks)
- CLI with configurable options
- Comprehensive documentation (README.md, AGENTS.md)
```

- Set up git repo
- Vibe coded PureData

```
feat: Add Pure Data integration for real-time sound synthesis

- Add PureDataSink (OSC) and PureDataFUDISink (TCP) output sinks
- Add --pd-osc and --pd-fudi CLI flags with host/port options
- Add python-osc dependency
- Create Pure Data patches:
  - brightness_simple.pd: No externals needed, uses FUDI/TCP
  - brightness_receiver.pd: Uses OSC (requires mrpeach external)
- Add test script for verifying Pd connection without Pupil hardware

```

- We spent a lot of time calibrating the glasses in Pupil Capture today. The automatic one didn’t work for Jørgen’s mac, and it was a bit bad for me. We ended up doing it manually with picking points in the environment (”natural point” option). Jørgen would click them on the screen in while I was looking. That worked a lot better.
- We thought about how to do calibration in an actual setting. Maybe we need to have a custom script that takes custom markers that we can place around a room. Or a person walks around with one? Sound as confirmation.

### 28.12.25

- Installed Claude Code and set it up with VSCode
- Started looking for literature for mapping

### 29.12.25

Time spent: ~3h

- Started developing code for color mapping → changed so that color maps to note and brightness map to octave
    - Orange not registering properly
    - Too jumpy and not registering color and brightness well enough
    
    ![image.png](Project%20Log/image.png)
    
- One of the eye-cameras suddenly went dark (very under-exposed to the point where the eye is not visible)

### 30.12.25

Time spent: ~3h

- Kept prototyping using only one eye-camera
- Made a test sheet for colors with brightness. Values are correct into pure data. Real life application is harder, but is starting to make more sense. The problem is the accuracy of the gaze and the huge variation in hue and brightness in everyday objects and environments. Would be beneficial to test with different region sizes. Need to record or preferably have another person present to check in real-time.
    - uv run pupil-tracker --pd-color-fudi --region-size 80
- Does it matter? The point is to generate interesting and beautiful sound, not clinical accuracy in color and brightness detection.
- Had Claude do research into better techniques for color identification
- Very surprised and pleased by the abilities of Claude Code (Opus 4.5)

[Research Summary: Color Detection in Video Streams](https://www.notion.so/Research-Summary-Color-Detection-in-Video-Streams-2d9693cc579c8084ac27f6765b5e1332?pvs=21)

- Should the brightness range for the lowest and highest octave be a lot smaller maybe?

### 31.12.25

- Ran out of tokens yesterday. Looked over and committed changes today.
- Need to figure out a way to better test new functionality so that I can monitor it. The best would be to record video and gaze data from the glasses and then use those. Then I don’t need to have the glasses to keep working either.
    - Have to figure out how to get both world camera video and gaze data and feed them to the program
- I’m planning on using chatGPT to do some thorough research into what sort of ML models or other stuff I can use for object recognition and identification. The goal is to figure out how far away the object I am looking at is, and what the surface texture is. Maybe size would also be interesting, but not absolutely needed. It could be an alternative to brightness when it comes to the octave.

### 16.01.26

Time spent: ~1h?

- I had filmed some videoes with the glasses, unfortunately bad accuracy due to only being able to use one eye camera. Had Claude make a video player for the videoes and started trying to do some simple object recognition with YOLO

### 25.01.26

- Reverted YOLO, not sure if it’s necessary to recognize objects, and generic objects in general won’t do much for sound I think
- Gonna work on surface/material instead for timbre
- Modified video player to send to pd using a flag. Adds overlay to video player (color, note, brightness) as well

### 26.01.26

- Cleaned up. Removed OSC for pd, only using FUDI anyways, and that’s not dependent on external libraries
- Made the overlay in video player optional even if not sending to pd
- Decided to start keeping transcripts from the agent sessions. should have done this from the start..

### 28.01.26

Time: ~2h?

- Lots of smaller stuff.. cleaned up some shit.
- Added gamma correction to video feed
- Went over the pd patch again, added fade out when test/play is stopped, so it doesn’t freeze on the last note.

### 29.01.26

- Meeting with Jensenius
    - Evolved my sense of scope, mind blown type beat..

### 31.01.26

- Research into Granular Synthesis
- Started looking at literature

### 02.02.26

- Meeting with Maham
    - Mind patching up again :p
- Huge revision of project → intention, direction, work-flow, plan, paper
- Made a plan for a sort of gameified experience, level system to add complexity and promote exploration
    
    [Game Design (DEPRECATED)](https://www.notion.so/Game-Design-DEPRECATED-2fb693cc579c80a39f08c9388fe84690?pvs=21)
    

### 03.02.26

- Read about Oculog. It only looks at the eyes, not at the “world”, but it could be relevant to cite
- Reflected a bit about the next implementation steps
    - Melody:
        - Play only a short note when something is looked at
            - We need to designate a time limit for how long the gaze is held before the note is played
                - It should be short enough so that it feels responsive, but if it’s too short, it’s gonna be messy
                - An option is to have a short limit, but with a delay afterwards, so that it doesn’t jump to the next point immediately
                - Needs thorough manual testing to see what works best
            - We need to set an area around the gaze position where the exact middle point needs to stay in, to allow for minor flickering of either the eye or the tracking stability/confidence
            - We’ll shape the sound of the note in Pure Data so that it sounds nice
                - Sine wave or subtractive synthesis on a square wave or something
                - Reverb!!!!
    - Samples:
        - Train a very specific ML algorithm to recognise a few objects
            - How do I do this?
            - Should the object be recognisable from behind, or just front/sides?
                - Intuitively it would feel better if it’s looked “in the face” → feel like you are actually interacting with the object and not just spotting it
                - Could the “just spotting it” be exploited to have some sort of hint of the sample? Like “hey, there is something over there” in the *audio domain* and not just the visual one?
        - Have recognition of these objects trigger pre-made samples in Pure Data
            - Should the samples play all the way out?
            - Should they fade out when the object is looked away from?
                - If so → *continue* or *restart* on next look?
- I still have no idea how to do the sound that is evolving, but I’m thinking I’ll cross that bridge when I get there. I’ll do these two things first, and probably have something cool
- Another idea that popped up today is to have blink frequency do something. maybe I could do AM or something else that has a transformative effect on the sound in a way that is instantly and intuitively recognisable to the user and the audience. maybe if you blink fast enough for long enough that triggers an easter egg as well

### 14.02.26

![image.png](Project%20Log/image%201.png)

- I spent HOURS on trying to calibrate the glasses, and they would just not work accurately. Finally got a decent test recording (005), but it only worked when basing the gaze on eye1.
- I finally went in and got hands-on on the Pure Data patch. Claude is not great at it, and it was really simple to just do it myself.
- HOWEVER, the note events don’t trigger where you would expect in the video I took of looking at a spread of different colored books (005). This is a task for tomorrow, I think it will take some time to figure out when the note should play so that it feels natural, and to stop it from repeating itself. Should probably read about mapping tomorrow as well, but I think we can separate the exact mapping from the note triggering event itself.

### 15.02.26

- it’s 1 am.. anyways..

![image.png](Project%20Log/image%202.png)

![image.png](Project%20Log/image%203.png)

- It now works well enough on this video for us to move on to the next stage, I think? I should try it live with the glasses, but I’m too lazy to put the books out, and the lighting is terrible now anyways..
- Started on the blink detection later in the day. Claude has done some research, we’ll see how it works out.

### 16.02.26

- Continuing work on blink detection. Implemented using the pupil labs core blink detection event thing yesterday, but comparing that to a homecooked version rn. There are some issues with noise, predictably. Might be worth it to try and get as clean as possible readings on the “now the eye is closed” part and accept the noise as a feature maybe.
- The blink detection implementation got squashed into the flutter detection commit. Wops.
    - I asked Claude to fix this. It worked lmao
- Did some research on specific object detection. Thoughts:
    - We need an effective way to train the model, aka effectively take and mark images
        - Since we have the eye-tracking up and running with a video feed, we should use this! Maybe we set up a flag on the live eye-tracking that turns off all of our functionality, and instead draws a big box with a label around the gaze position. We then look at the objects, make sure they are inside the box, and then click space to save the image. This way we can take many images from many angles and in different settings and lighting effectively. We should be able to write/change the label of the thing we are photographing while we are working, to seamlessly move on to the next thing, and every image taken with a certain label should be put in a corresponding folder.

### 18.02.26

- I think I’m scrapping melody mode/exploration mode and only doing melody mode but with light shimmering in the background or something. like hints. to do this, I gotta think about how to use eye qualities for musical purposes. Changing the project plan.

### 23.02.26

- We need to make sure all functionality that works with the video player also works in real-time
- I’m having trouble with the notes not triggering correctly. There was an issue before when the gaze moved between objects with the same color and brightness, it wouldn’t re-trigger the note. This got a lot worse when I removed the lower octave and gave the middle octaves more space. Tried to fix it with fixation detection (didn’t work) and then with detecting gaps in MIDI-notes before smoothing, but not making any difference. Not sure if I should just reset the octave mappings for now and accept the errors.
    - Ended up resetting the ocatave mappings, but combined octave 2 and 3
- Now suppressing notes during flutters

### 24.02.26

- Calibration is SO FRUSTRATING. It’s impossible to play anything when the tracker is a little bit off or sometimes just bouncing all over the place
    - I need to see if the bouncing all over the place happens in the video player as well

### 25.02.26

- Changed the pd patch to add AM. python code now sends midi to am_lfo to change the freq based on the number of blinks in a flutter with a max of 15 to freq. 50 hz. an intentional/long blink resets the value
    - I considered making the am freq decay over time, but I think the user experience is better if you can change the value and then reset it when you want and potentially re-do it. gives more control, probably feels more intuitive. also good if you want to keep it for an extended period of time
- A few small things:
    - The volume is pretty low now when the LFO freq is 0? → I had to restart pd..
    - We need to reset it to 0 on close/start. Idk what’s best. → fixed along with dist. signal volume
    - It doesn’t do that much cool stuff rn with the short notes being played. Do we add an even longer reverb, or should we just do it like this and have the effect be more noticable when looking at stuff for longer (the evolving sound)? It would def. be cool if it did have an effect on that.
        - I added a longer reverb which makes it easier to hear the AM. It can be for demo purposes maybe, because it feels kinda weird in the room.
- Added noise signal for blinks/closed eyes/flutter, this immediately felt really fun and exciting!! it’s just generic pd noise~ for now, but obv this can be modified and expanded upon

### 05.03.26

- I sort of forgot that I was keeping a log, but I’ve mostly been writing anyways.
- Did start with granulary synthesis, finally
    - I kind of want the sample to play almost normally first and then start to fragment, and then just fragment more and more and go wild/random
    - I have a pd patch rn but I don’t know yet how to do the evolving fragmentation

### 06.03.26

- Some thoughts on game elements:
    - what if after you long-look or even short-look at all the different colours, you unlock something new? like a new level
    - Play short melody → get sample
        - Think this would work best if there were melodies/motifs/themes that are very close to each other, simple, might trigger accidentally even
            - Ja vi elsker..
            - la fille aux cheveux de lin
- After meeting with Balint..
    - I think maybe I should de-couple the delevopment and the thesis a bit. Like, I’m going to keep developing the whole thing, but I think I’ll focus on writing only about the eye-specific parts.

### 23.03.26

- I feel so lost atm. I haven’t done anything in over 2 weeks, except some writing. I think the lack of progression on the development is really demotivating, because I feel like I’m not getting anywhere and have no idea what my thing actually is. Because it’s not really anything. Well, it’s *something,* but not anywhere near where I wanted it to be. It’s not just that I have ideas that I haven’t implemented yet, it’s more that I’m conceptually lost. What am I even making? Is it an instrument? Do I want people to play it? Is it trying to do too many different things? Probably, but that wouldn’t be an issue in an exploratory experience. In an instrument, however.. it would. But I didn’t really set out to create a rigid instrument in the first place. So why is that where I feel like I have to go?
- I think it would be very helpful if I started up the practical aspect again, but the calibration troubles feel like such a huge mountain to climb every time I think of setting things up to try. What does it say about my thing, if I’m not even excited enough to use it to get past the callibration issues?
- It’s also a huge issue that I need to work on it at home if I’m going to do stuff live with the glasses. I’m only in work mode in the library these days. I can’t get into it at home for some reason.

### 25.03.26

- I’m currently testing the system in practice (pupil-tracker)
- I think maybe because of the limitations of the hardware, I need to think differently about this. Maybe give up on using the gaze point as a precise control element. I need to think about it as a bit more volatile. Should I change note playing as the central functionality, or keep it as is, or think about it more as changing the note palette by looking at different “collections” of colours?
- I also think I need to expand my perspective on affordances, and start also thinking about what the affordances of what is being looked at are/can be.
- I’m noticing that my eyes are getting really tired after maybe 30-45 minutes of this. It might be because I usually need glasses, even though it’s not really that bad for me. But I think it might also mainly be because I’m tired and woke up only about an hour before starting the testing (update: I’m probably getting a cold)

### 26.03.26

- After talking to Jensenius yesterday, I understood that there is a lot of value in the ground work I’m doing of setting up the glasses for musical use. I’ve decided to do a solid job with this before moving on to the creative part of the mapping so I can make a lot of different prototypes. This also has the advantage of creating a system that other people can build upon, to do their own mapping.
- This naturally lead to a huge refactor of the code repo. Structuring the code so that the paramteres to map and the actual mapping is separated. I’m also planning to make it clearer the two possible inputs to the pipeline: live and recording playback. I think both of these are good when it comes to music making. The output sinks are supposed to be separate as well. I’ve yet to test if the values created from each parameter is a good representation, but for now everything is distilled to simple numeric values, coordinates or bools. The idea is that the only thing you need to do to change the creative mapping is to plug in a new python file that defines prototype-specific logic (if needed?), and connect the sound-producing file (pure data, Max, anything really) to the sinks.
- I’m spending a lot of time making this clean and strong.

### 28.03.26

- Started working on a new pd patch “ocean”
- Had chatGPT suggest some ways to use a fibonacci sequence to mess with a base frequency. Ended up with this one:

![image.png](Project%20Log/image%204.png)

- Added two metronomes to trigger notes at increasing tempo

![image.png](Project%20Log/image%205.png)

### 29.03.26

- Did some small fixed to the base code, but I think everything is ready to just start prototyping like hell at this point
- Vibe coding the python side of the patch I started working on yesterday. Just gonna use colours again, I think, but mapping 1-7 for base frequency scaling. So it’s the same as the other one, I’m just doing the MIDI logic in pure data because I need to retrigger it all the time with the bangs to process the expression object
- Testing this later in the tracker. None of the videos have a long enough dwell to get a good idea of the use case.
- Noticed another small thing in the base code, the closing pd messages were coded into [player.py](http://player.py) and [tracker.py](http://tracker.py) even though they’re patch specific. Pretty quick to fix, but I wonder what else Claude didn’t notice when refactoring. Will keep an eye out and fix as I go.

### 05.04.26

- Creating a naming convention for the prototypes.
- I need to make a prototype that uses continuous streams of data
- I just want to quickly note that after the last conversation with Jensenius, I have gone from an iterative approach to a sort of.. branched one? Where I have a base and make prototypes from it and reflect around the prototypes. Quantity not quality but not really? Idk.

### 06.04.26

- Implementet (most of) SCf-v1
    - Should I do something with blinks? It works as it is, so I’ll come back to it later
    - Still have to decide if I want it to be inverted or not. Maybe I’ll made v2 where the ambience comes from closing the eyes
- Refactored overlay. Patches can now request what overlay they want individually.
    - Another thing Claude didn’t notice in the old, huge refactor: tracker and player was drawing stuff and handling overlay logic. Should be in overlay.py, and tracker and player should just call the overlay. this is fixed now.

### 08.04.26

- Connected the glasses to RAVE model. Fun. The model itself is a bit confusing, but you could hear after carefully selecting parameters and mappings the control from the eyes
- Decided to implement separate detection for both eyes when it comes to closures and blinks.
    - I’m dumb, because I need to be writing/reading for the thesis rn, and this is probably going to take a while to iron out. Should put a pin in it and continue prototyping tomorrow, but I get too caught up in these things. Overwhelming. Doing too much at once right now.
- Got inspired to make a new patch where the gaze position decides pitch and loudness. Based on metaphores from lakoff! And experiences from RAVE Hackaton, the most controllable parameteres (that I went for at least) was the x and y position, as well as intentional eye closure.
- If I’m going to be collecting experiences from other people, I need to finish and structure the prototypes so that I can just pull the whole thing up and let people try. I need to ready the question sheet as well.

### 17.04.26

- Lowkey gave up on having other people test. Not enough time, will fry myself
-