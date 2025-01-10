var colors = [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf'
]

var colors2 = [
    '#E27D60',
    '#41B3A3',
    '#E8A87C',
    '#C38D9E',
    '#7AE7C7'
]

var state = {};

state.unlocked = false;
state.token = undefined;
state.token = getCookie('token');
state.modal = document.getElementById('defaultKeyboardShortcuts')
if (state.token) {
    var url = '/get-token/' + state.token;
    fetch(url)
        .then(response => response.json())
        .then(valid => {
            console.log(valid.valid);
            if (valid.valid) {
                state.unlocked = true;
                state.modal = document.getElementById('keyboardShortcuts');
                console.log('unlocked')
            }
            drawButtons();
        });
} else {
    drawButtons();
}

// var modal = document.getElementById('keyboardShortcuts');
var keyboardShortcutsButton = document.getElementById('keyboardShortcutsButton');
var spanLocked = document.getElementsByClassName('close')[0];
var spanUnlocked = document.getElementsByClassName('close')[1];

keyboardShortcutsButton.onclick = function() {
    state.modal.style.display = 'block';
}

spanLocked.onclick = function() {
    state.modal.style.display = 'none';
}

spanUnlocked.onclick = function() {
    state.modal.style.display = 'none';
}

window.onclick = function(event) {
    if (event.target == state.modal) {
        state.modal.style.display = 'none';
    }
}

window.addEventListener('DOMContentLoaded', function(){
    // get the canvas DOM element
    var canvas = document.getElementById('renderCanvas');

    // load the 3D engine
    state.engine = new BABYLON.Engine(canvas, true);

    // createScene function that creates and return the scene
    var createScene = function() {

        var scheme = undefined
        var keypoints = undefined;

        // Create the scene space
        var scene = new BABYLON.Scene(state.engine);

        var ambiance = 0.5;
        scene.ambientColor = new BABYLON.Color3(ambiance, ambiance, ambiance);


        // // Add a camera to the scene and attach it to the canvas
        var camera = new BABYLON.ArcRotateCamera(
            "Camera",
            0, 0, 10,
            BABYLON.Vector3.Zero(),
            scene
        );
        camera.setPosition(new BABYLON.Vector3(0, 0, -10));
        camera.fovMode = 0;

        // // Parameters: name, position, scene
        // // var camera = new BABYLON.FlyCamera("FlyCamera", new BABYLON.Vector3(0, 0, 0), scene);
        // var camera = new BABYLON.UniversalCamera("UniversalCamera", new BABYLON.Vector3(0, 0, -6), scene);

        camera.attachControl(canvas, true);
        camera.lowerBetaLimit = null;
        camera.wheelPrecision = 50;
        
        // Add lights to the scene
        var light1 = new BABYLON.HemisphericLight(
            "light1",
            new BABYLON.Vector3(0, 1, 1),
            scene
        );
        light1.intensity = 0.3;

        // var light2 = new BABYLON.PointLight(
        //     "light2",
        //     new BABYLON.Vector3(0, 1, -1),
        //     scene
        // );
        // light2.intensity = 0.2;

        var light3 = new BABYLON.HemisphericLight(
            "light3",
            new BABYLON.Vector3(-1, -1, 0),
            scene
        );
        light3.intensity = 0.3

        var scale = 3;

        drawSpheres(scene, keypoints, scale);
        drawTubes(scene, scheme, keypoints, scale);

        state.scene = scene;

        return scene;
    };

    // call the createScene function
    var scene = createScene();
    var divFps = document.getElementById("fps");

    state.allBehaviorChanges = {};
    state.behaviorChanges = [];
    state.filterBehavior = '';
    var selectBehavior = document.getElementById("selectBehavior");
    var actogram = document.getElementById("actogram");
    var vidlist = document.getElementById('vidlist');

    // run the render loop
    state.engine.runRenderLoop(function() {
        scene.render();
        divFps.innerHTML = state.engine.getFps().toFixed() + " fps"; 
    });

    // the canvas/window resize event handler
    window.addEventListener('resize', function() {
        state.engine.resize();
    });

    var progressBar = document.getElementById("progressBar");
    progressBar.addEventListener(
        "mousedown", function(e) { setPlayPosition(e.pageX); },
        false);

    $(document).keyup(function(e) {
        if (e.keyCode==187) { // +
            speedupVideo();
        } else if (e.keyCode==189) { // -
            slowdownVideo();
        } else if (e.keyCode==190) { // .
            advanceFrame(1);
        } else if (e.keyCode==188) { // ,
            advanceFrame(-1);
        }
    });

    $(document).keyup(function(e) {
        if (e.key === "Escape") { 
            state.selectedBout = undefined;
            state.selectedBehavior = undefined;
            drawActogram();
        }
    });

    $(document).mouseup(function(e) {
        if (state.selectedBout) { 
            state.selectedBout = undefined;
            state.selectedBehavior = undefined;
            drawActogram();
        }
    });

    window.addEventListener('keydown', function(e) {
        if(e.keyCode == 32 && e.target == document.body) {
            e.preventDefault();
            togglePlayPause();
        }
    });

    window.addEventListener("keydown", function(e) {
        // up and down arrow keys
        if (e.keyCode == 38  || e.keyCode == 40) {
            e.preventDefault();
        }
    }, false);


    // state.trial = {
    //     session: "5.16.19",
    //     folder: "Fly 2_0",
    //     files: [
    //         "05162019_fly2_0 R1C1 Cam-A str-cw-0 sec",
    //         "05162019_fly2_0 R1C1 Cam-B str-cw-0 sec",
    //         "05162019_fly2_0 R1C1 Cam-C str-cw-0 sec",
    //         "05162019_fly2_0 R1C1 Cam-D str-cw-0 sec",
    //         "05162019_fly2_0 R1C1 Cam-E str-cw-0 sec",
    //         "05162019_fly2_0 R1C1 Cam-F str-cw-0 sec"
    //     ],
    //     vidname: "05162019_fly2_0 R1C1  str-cw-0 sec"
    // }

    // decode the url
    var h = decodeURIComponent(window.location.hash.substring(1));
    var L = h.split("/");
    var state_url = {};
    if(L.length == 3) {
        state_url.session = L[0];
        state_url.folder = L[1];
        state_url.trial = L[2];
    }
    console.log(state_url);

    fetch('/get-sessions')
        .then(response => response.json())
        .then(data => {
            console.log(state_url);
            state.sessions = data.sessions;

            $('#selectSession').empty();
            var list = $('#selectSession');
            for(var num=0; num<data.sessions.length; num++) {
                var session = data.sessions[num];
                list.append(new Option(session, session));
            }

            var ix = state.sessions.indexOf(state_url.session);
            if(ix != -1) {
                list.val(state_url.session);
                updateSession(state_url.session, state_url);
            } else {
                updateSession(data.sessions[0]);
            }
        })

    $('#selectSession').select2({
        matcher: matcher
    });

    $('#selectVideo').select2({
        matcher: matcher
    });

    $('#selectBehavior').select2({
        matcher: matcher
    });

    $('#selectSession').on('select2:select', function(e) {
        var d = $('#selectSession').select2('data');
        var session = d[0].id;
        updateSession(session);
    });

    $('#selectVideo').on('select2:select', function (e) {
        var d = $('#selectVideo').select2('data');
        var trial = state.trials[d[0].id];
        updateTrial(trial);
    });

    $('#selectBehavior').on('select2:select', function (e) {
        var d = $('#selectBehavior').select2('data');
        state.filterBehavior = d[0].id;
        filterTrials();
    });

    updateSpeedText();

});

function matcher(params, data) {
    if ($.trim(params.term) === '') {
        return data;
    }

    keywords=(params.term).split(" ");

    for (var i = 0; i < keywords.length; i++) {
        if (((data.text).toUpperCase()).indexOf((keywords[i]).toUpperCase()) == -1)
            return null;
    }
    return data;
}

function filterTrials() {

    console.log(state.trials)
    var ixs = [];
    state.videoIndexes = {};
    $('#selectVideo').empty();
    var filteredTrials = $("#selectVideo");
    for (var j in state.trials) {
        var trial = state.trials[j]
        var rel_path = trial.session + '/' + trial.folder + '/' + trial.vidname;
        if (state.filterBehavior == "" ||
            (state.possible.trialBehaviors[rel_path] !== undefined && state.possible.trialBehaviors[rel_path][state.filterBehavior])) {
            var text = trial.vidname + " -- " + trial.folder;
            var key = j + "";
            ixs.push(j)
            filteredTrials.append(new Option(text, key))
            state.videoIndexes[rel_path] = parseInt(j)
        }
    };
    updateTrial(state.trials[ixs[0]]);
}

function nextVideo() {
    var url_suffix = state.trial.session + "/" + state.trial.folder + "/" + state.trial.vidname;
    var allIndexes = Object.values(state.videoIndexes)
    var currentIndex = allIndexes.indexOf(state.videoIndexes[url_suffix])
    if (currentIndex < allIndexes[allIndexes.length-1]) {
        var newIndex = allIndexes[currentIndex + 1]
        updateTrial(state.trials[newIndex]) 
        $('#selectVideo').val(newIndex).change()
    }
}

function previousVideo() {
    var url_suffix = state.trial.session + "/" + state.trial.folder + "/" + state.trial.vidname;
    var allIndexes = Object.values(state.videoIndexes)
    var currentIndex = allIndexes.indexOf(state.videoIndexes[url_suffix])
    if (currentIndex > 0) {
        var newIndex = allIndexes[currentIndex - 1]
        updateTrial(state.trials[newIndex])
        $('#selectVideo').val(newIndex).change()
    }
}

function clickVideo(e) {
    console.log(e);
    var container = e.target.parentElement;
    console.log(container);
    var parent = container.parentElement;
    var vidlist = document.getElementById("vidlist");
    var vidlistUnfocused = document.getElementById("vidlistUnfocused");

    if(parent == vidlist) {
        vidlist.removeChild(container);
        vidlistUnfocused.appendChild(container);
    } else {
        vidlistUnfocused.removeChild(container);
        vidlist.appendChild(container);
    }
}

function updateSession(session, state_url) {

    state.metadata = undefined;
    fetch('/metadata/' + session)
        .then(response => response.json())
        .then(data => {
            state.metadata = data;
        });

    document.getElementById('actogram').innerHTML = '';
    state.behaviorList = undefined;
    state.trials = undefined;
    state.trial = undefined
    state.possible = undefined;
    fetch('/get-trials/' + session)
        .then(response => response.json())
        .then(data => {
            state.possible = data;
            state.session = data.session;
            state.trials = [];

            vidlist.innerHTML = '';
            var ncams = data.folders[0].files[0].camnames.length; 
            for (var i = 0; i < ncams; i++) {

                var container = document.createElement("div");
                container.className = "container";
                vidlist.appendChild(container);

                var video = document.createElement("video");
                video.className = "vid";
                video.preload = "auto";
                video.loop = true;
                container.appendChild(video);

                var canvas = document.createElement("canvas");
                canvas.className = "canvas";
                container.appendChild(canvas);

                container.onclick = clickVideo;
            }

            $('#selectBehavior').empty();
            var behaviorList = $("#selectBehavior");
            behaviorList.append(new Option('', ''));
            data.sessionBehaviors = data.sessionBehaviors.sort();
            for (var i in data.sessionBehaviors) {
                behaviorList.append(new Option(data.sessionBehaviors[i], data.sessionBehaviors[i]));
            }
            behaviorList.val("");

            var ix = 0;
            state.videoIndexes = {};
            $('#selectVideo').empty();
            var list = $("#selectVideo");
            var vidname_folder_ix = {};
            for(var folder_num=0; folder_num < data.folders.length; folder_num++) {
                console.log(folder_num);
                var folder = data.folders[folder_num]; 
                for(var file_num=0; file_num < folder.files.length; file_num++) {
                    var file = folder.files[file_num];
                    file.session = data.session;
                    file.folder = folder.folder;
                    var text = file.vidname + " -- " + file.folder;
                    var key = ix + "";
                    vidname_folder_ix[text] = key;
                    state.trials[key] = file;
                    list.append(new Option(text, key));
                    var url_suffix = state.trials[key].session + "/" + state.trials[key].folder + "/" + state.trials[key].vidname;
                    state.videoIndexes[url_suffix] = ix;
                    ix++;
                }
            }

            state.trial = state.trials[0];
            var key = "0";
            if(state_url) {
                var text = state_url.trial + " -- " + state_url.folder;
                var test = vidname_folder_ix[text]
                if(test) {
                    key = test;
                }
            }

            updateTrial(state.trials[key]);
            list.val(key);;

        });

}

function updateTrial(trial) {

    var url_suffix = state.trial.session + "/" + state.trial.folder + "/" + state.trial.vidname;
    for (var i=0; i<state.behaviorChanges.length; i++) {
        if (!state.allBehaviorChanges[url_suffix]) {
            state.allBehaviorChanges[url_suffix] = [];
        } 
        state.allBehaviorChanges[url_suffix].push(state.behaviorChanges[i])
    }
    state.behaviorChanges = [];
    state.redo = [];

    console.log(trial);
    state.trial = trial;
    var url_suffix = trial.session + "/" + trial.folder + "/" + trial.vidname;
    console.log(url_suffix)
    window.location.hash = "#" + url_suffix;
    state.camnames = trial.camnames;

    var nextButton = document.getElementById('nextVideo');
    var previousButton = document.getElementById('previousVideo');
    var ix = state.videoIndexes[url_suffix];
    var allIndexes = Object.values(state.videoIndexes);
    var currentIndex = allIndexes.indexOf(ix);
    if (ix == allIndexes[0]) {
        previousButton.style.visibility = 'hidden';
    } else if (ix == allIndexes[allIndexes.length-1]) {
        nextButton.style.visibility = 'hidden';
    } else {
        previousButton.style.visibility = 'visible';
        nextButton.style.visibility = 'visible';
    }

    // if (state.allBehaviorChanges[url_suffix]) {
    //    state.behaviorChanges = state.allBehaviorChanges[url_suffix];
    // }

    playing = false;
    hide2d = false;
    updateSpeedText();
    updatePlayPauseButton();
    updateToggle2DButton();
    updateToggle3DButton();

    var container3d = document.getElementById("modelContainer");
    if (!display3d) {
        container3d.classList.add("hidden");
    } else {
        container3d.classList.remove("hidden");
    }


    var url;
    url = '/pose3d/' + url_suffix;
    fetch(url)
        .then(response => response.json())
        .then(data => {
            console.log(data)
            console.log("pose 3d updated");
            state.data = data;
            updateKeypoints(data[0]);
            drawFrame(true);
        });

    url = '/pose2dproj/' + url_suffix;
    state.data2d = undefined;
    fetch(url)
        .then(response => response.json())
        .then(data => {
            console.log("pose 2d updated");
            state.data2d = data;
            drawFrame(true);
        });

    var videoContainer = document.getElementById("videoContainer");

    // we sort like this so we can preserve the location of videos as arranged by user
    state.videos = videoContainer.querySelectorAll("video");
    state.videos = state.videos.values().toArray();
    state.videos.sort(function(a, b) {
        if(a.src < b.src) {
            return -1;
        } else if(a.src > b.src) {
            return 1;
        } else {
            return 0;
        }
    });

    state.containers = state.videos.map(function(x) { return x.parentElement; });
    state.canvases = state.containers.map(function(x) { return x.querySelector("canvas"); });

    // state.canvases = videoContainer.querySelectorAll("canvas");
    // state.containers = videoContainer.querySelectorAll(".container");
    state.videoLoaded = false;
    state.behaviorLoaded = false;

    for(var i=0; i<state.videos.length; i++) {
        var video = state.videos[i];
        var url = "/video/" + trial.session + "/" + trial.folder + "/" + trial.files[i];
        video.src = url;
        console.log(url);
    }


    for(var i=0; i<state.canvases.length; i++) {

        var vid = state.videos[i];
        vid.index = i;

        vid.addEventListener("loadeddata", function(e) {
            var i = this.index;
            var width = this.clientWidth;
            var height = this.clientHeight;
            console.log(width, height);

            var canvas = state.canvases[i];
            var ctx = canvas.getContext("2d");
            var container = state.containers[i];

            ctx.canvas.width = width;
            ctx.canvas.height = height;

            state.containers[i].style.width = width +"px";
            state.containers[i].style.height = height + "px";

            state.engine.resize(); 

            if(i == 0) {
                updateProgressBar();
                console.log('video loaded');
                state.videoLoaded = true;          
                if (state.behaviorLoaded) {
                    if (state.allBehaviorChanges[url_suffix]) {
                        applyBehaviorChanges();
                        state.uniqueTrialBehaviors = getUniqueTrialBehaviors();
                    }
                    drawActogram();
                }
            }
        }, false);
    }

    var url = '/framerate/' + trial.session + "/" + trial.folder + "/" + trial.files[0];
    state.fps = undefined;
    fetch(url)
        .then(response => response.json())
        .then(data => {
            state.fps = data;
            if(state.videoLoaded && state.behaviorLoaded) {
                console.log('drawing actogram again')
                drawActogram();
            }
        });

    // state.videos[0].addEventListener('timeupdate', updateProgressBar, false);
    setInterval(function () {
        // console.log(state.videos[0].currentTime);
        updateProgressBar();
    }, 10);

    setInterval(function () {

        if (!state.metadata) {
            return
        };

        var totalmseconds = Math.floor(state.videos[0].duration * 1000);
        var currentmseconds = Math.floor(state.videos[0].currentTime * 1000);
        timer.innerHTML = formatTime(currentmseconds) + ' / ' + formatTime(totalmseconds);

        var currFrame = updateFrameNumber();
        var nFrames = Math.floor(state.videos[0].duration * state.fps)
        frameCount.innerHTML = currFrame + ' / ' + nFrames;

    }, 5);
    
    url = '/behavior/' + url_suffix;
    state.behaviors = undefined;
    state.uniqueTrialBehaviors = undefined;
    fetch(url)
        .then(response => response.json())
        .then(data => {
            console.log("behavior updated");
            state.behaviors = data;
            state.uniqueTrialBehaviors = getUniqueTrialBehaviors();
            state.behaviorLoaded = true;
            if (state.videoLoaded) {
                console.log('drawing actogram')
                if (state.allBehaviorChanges[url_suffix]) {
                    applyBehaviorChanges();
                    state.uniqueTrialBehaviors = getUniqueTrialBehaviors();
                }
                drawActogram();
            }
        });



}

function downloadBehaviors() {

    var url = '/download-behavior/' + state.session;
    fetch(url)
        .then(response => response.json())
        .then(data => {
            download(data);
        });
}

function download(data) {
    var behaviors_json = new Blob(
        [JSON.stringify(data)], 
        {type: 'text/json;charset=utf-8'}
    );
    var session_name = state.session;
    var url = URL.createObjectURL(behaviors_json);
    var anchor = document.createElement('a');
    anchor.href = url;
    anchor.target = '_blank';
    anchor.download = session_name + '_behaviors.json';
    anchor.click();
    URL.revokeObjectURL(url);
}

function formatTime(milliseconds) {
    milliseconds = state.metadata.video_speed * milliseconds;
    var mseconds = Math.floor(milliseconds % 1000)
    mseconds = mseconds.toString().substring(0, 2);
    mseconds = (mseconds >= 10) ? mseconds : '0' + mseconds;
    var seconds = Math.floor(milliseconds / 1000) % 60;
    seconds = (seconds >= 10) ? seconds : '0' + seconds;
    var minutes = Math.floor(milliseconds / 1000 / 60);
    minutes = (minutes >= 10) ? minutes : '0' + minutes;
    return minutes + ':' + seconds + ':' + mseconds;
}

var slowdown = 1;
// var rate_estimate = state.vid_fps/state.fps*slowdown;
var framenum = 0;
var playing = false;
var display2d = false;
var display3d = false;
var prev_num = 0;

function drawFrame(force) {
    if(!playing && !force) return;

    var ft = state.videos[0].currentTime * state.fps;
    // var diff = ft - framenum;
    // if(ft > 5) {
    //     rate_estimate = 0.9 * rate_estimate + 0.1 * diff;
    // }
    // if(Math.abs(diff) > 6) {
    //     framenum = ft;
    // } else {
    //     framenum += rate_estimate;
    // }
    // prev_num = ft;
    // console.log(ft);
    // console.log(rate_estimate, framenum, ft, ft - framenum);
    // if(Math.abs(ft - framenum) > 5) {
    //     framenum = ft;
    // }

    framenum = Math.round(ft+1);
    var nFrames = state.videos[0].duration * state.fps
    if (state.selectedBout) {
        if (framenum > state.bouts[state.selectedBehavior][state.selectedBout].end) {
            for (var i = 0; i < state.videos.length; i++) {
                state.videos[i].currentTime = (state.bouts[state.selectedBehavior][state.selectedBout].start / nFrames) * state.videos[0].duration;
            }
        }
    }

    if(state.data) {
        const fix = Math.max(0, Math.min(Math.floor(framenum), state.data.length-1));
        setTimeout(function() {
            updateKeypoints(state.data[fix])
            draw2D(fix);
        }, 0);
    }
    if(playing) {
        setTimeout(drawFrame, 1000.0/state.fps);
    }
    // window.requestAnimationFrame(drawFrame);
}


function drawNextFrame(force, framenum) {
    if(!playing && !force) {
        return;
    }
    var nFrames = state.videos[0].duration * state.fps
    for (var i = 0; i < state.videos.length; i++) {
        state.videos[i].currentTime = (framenum / nFrames) * state.videos[0].duration;
    }

    if(state.data) {
        const fix = Math.max(0, Math.min(Math.floor(framenum), state.data.length-1));
        setTimeout(function() {
            updateKeypoints(state.data[fix])
            draw2D(fix);
        }, 0);
    }
}

function getUniqueTrialBehaviors() {
    var uniqueTrialBehaviors = new Set();
    Object.keys(state.behaviors).forEach(function(id) {
        uniqueTrialBehaviors.add(state.behaviors[id]['behavior']);
    });
    var uniqueTrialBehaviors = Array.from(uniqueTrialBehaviors);
    return uniqueTrialBehaviors
}

// function getUniqueTrialBehaviors() {
//     var uniqueTrialBehaviors = new Set();
//     var laser_id = undefined; 
//     Object.keys(state.behaviors).forEach(function(id) {
//         if (state.behaviors[id]['behavior'] == 'laser') {
//             laser_id = id;
//         } else {
//             uniqueTrialBehaviors.add(state.behaviors[id]['behavior']);
//         }
//     });
//     var uniqueTrialBehaviors = Array.from(uniqueTrialBehaviors);

//     return uniqueTrialBehaviors
// }

function undo() {
    if (state.behaviorChanges.length === 0) {
        state.uniqueTrialBehaviors = Object.values(state.behaviorIds);
        drawActogram();
        alert('no changes to undo');
        return;
    }
    var change = state.behaviorChanges.pop();
    var video = state.trial.session + "/" + state.trial.folder + "/" + state.trial.vidname
    if (change.modification === 'changed behavior') {
        for (var i=0; i<state.behaviorChanges.length; i++) {
            if (!state.allBehaviorChanges[video]) {
                state.allBehaviorChanges[video] = [];
            } 
            state.allBehaviorChanges[video].push(state.behaviorChanges[i])
        }
        state.behaviorChanges = [];
        // Object.keys(state.behaviors).forEach(function(id) {
        //     if (state.behaviors[id].behavior === change.old.behavior) {
        //         state.behaviors[id].behavior = change.new.behavior;
        //     }
        // });
    } else if (change.modification !== 'added') {
        state.behaviors[change.id] = change.old;
    } else {
        delete state.behaviors[change.id];
    }
    state.redo.push(change);
    state.uniqueTrialBehaviors = Object.values(state.behaviorIds);
    drawActogram();
}

function redo() {
    if (state.redo.length === 0) {
        alert('no changes to redo');
        return; 
    }

    var change = state.redo.pop(); 
    if (change.modification === 'changed behavior') {
        state.redo = [];
        // Object.keys(state.behaviors).forEach(function(key) {
        //     if (state.behaviors[key].behavior === change.new.behavior) {
        //         state.behaviors[key].behavior = change.old.behavior;
        //     }
        // });
    } else if (change.modification !== 'removed') {
        var restoredBout = JSON.parse(JSON.stringify(change.old));
        Object.keys(change.new).forEach(function(key) {
            restoredBout[key] = change.new[key];
        });
        state.behaviors[change.id] = restoredBout; 
    } else {
        delete state.behaviors[change.id];
    }
    state.behaviorChanges.push(change);
    drawActogram();
}

function applyBehaviorChanges() {
    var video = state.trial.session + "/" + state.trial.folder + "/" + state.trial.vidname
    var changes = state.allBehaviorChanges[video]
    for (var i in changes) {
        var change = changes[i]
        if (change.modification === 'removed') {
            delete state.behaviors[change.id];
        } else {
            var restoredBout = JSON.parse(JSON.stringify(change.old));
            Object.keys(change.new).forEach(function(key) {
                restoredBout[key] = change.new[key];
            });
            state.behaviors[change.id] = restoredBout; 
        }
    }

}

function drawActogram() {

    actogram.innerHTML = '';
    console.log(state.behaviors);
    state.behaviorCanvases = {};
    state.behaviorIds = {};
    state.behaviorOrder = {};
    state.bouts = {};
    state.selectedBehavior = undefined;
    state.selectedBout = undefined;
    state.selectedFrameOffset = undefined;
    state.expectResize = -1;
    state.isResizeDrag = false;
    state.isDrag = false;
    state.modified = false;

    // state.uniqueTrialBehaviors = getUniqueTrialBehaviors();
    for (var i in state.uniqueTrialBehaviors) {
        var behaviorId = generateId(10);
        state.behaviorIds[behaviorId] = state.uniqueTrialBehaviors[i];
        Object.keys(state.behaviors).forEach(function(id) {
            if (state.behaviors[id].behavior === state.uniqueTrialBehaviors[i]) {
                state.behaviors[id].behavior_id = behaviorId;
            }
        });
    }

    var ix = 0;
    Object.keys(state.behaviorIds).forEach(function(behaviorId) {
        var behaviorContainer = document.createElement('div');
        behaviorContainer.className = 'behaviorContainer';
        behaviorContainer.style.height = '32px';
        actogram.appendChild(behaviorContainer);

        var color = colors2[i%colors2.length];

        var behaviorName = document.createElement('input');
        behaviorName.className = 'behaviorName';
        behaviorName.id = 'name' + ix.toString();
        behaviorName.value = state.behaviorIds[behaviorId]; 
        behaviorName.style.border = '1px solid ' + colors2[ix%colors2.length];
        if (!state.unlocked) {
            behaviorName.readOnly = true;
        }
        behaviorContainer.appendChild(behaviorName);

        var behaviorCanvas = document.createElement('canvas');
        behaviorCanvas.id = behaviorId;
        behaviorCanvas.className = 'behaviorCanvas';
        state.behaviorCanvases[behaviorCanvas.id] = behaviorCanvas;
        createBehavior(behaviorId, colors2[ix%colors2.length]);
        behaviorContainer.appendChild(behaviorCanvas);
        state.behaviorOrder[ix] = behaviorId;
        state.behaviorOrder[behaviorId] = ix;
        ix += 1;
    });

    console.log(state.behaviorOrder);

    var nFrames = state.videos[0].duration * state.fps;
    document.querySelectorAll('.behaviorCanvas').forEach(canvas => {
        var behaviorId = canvas.id;
        var ctx = state.behaviorCanvases[behaviorId].getContext("2d");

        state.behaviorCanvases[behaviorId].addEventListener('click', (e) => {
            var rect = state.behaviorCanvases[behaviorId].getBoundingClientRect();
            var point = {x: e.clientX - rect.left, y: e.clientY - rect.top};
            Object.keys(state.bouts[behaviorId]).forEach(function(key) {
                var bout = state.bouts[behaviorId][key];
                state.bouts[behaviorId][key].right = (rect.width-2) * (bout.end/nFrames);
                state.bouts[behaviorId][key].left = (rect.width-2) * (bout.start/nFrames);
                if (isSelected(point, bout)) {
                    if (state.selectedBehavior && state.selectedBout) {
                        var behaviorIdList = Object.keys(state.behaviorIds);
                        for (var i in behaviorIdList) {
                            createBehavior(behaviorIdList[i], colors2[i%colors2.length]);
                        }
                    }
                    state.selectedBehavior = behaviorId;
                    state.selectedBout = key;
                    state.selectedFrameOffset = Math.floor((point.x / (rect.width - 2)) * nFrames) - state.bouts[behaviorId][key].start;
                    selectBout(ctx);
                } else {                    
                    state.bouts[behaviorId][key].selected = false;
                    var sum = 0;
                    Object.keys(state.bouts[behaviorId]).forEach(function(key) {
                        sum += state.bouts[behaviorId][key].selected;
                    });
                    if (sum < 1 && state.selectedBehavior === behaviorId) {
                        state.selectedBehavior = undefined;
                        state.selectedBout = undefined;
                    }
                    ctx, color = getBoutColor(ctx, bout, behaviorId);
                    ctx.beginPath();
                    ctx.fillStyle = color;
                    if (state.unlocked) {
                        if (state.behaviors[bout.bout_id].manual) {
                            ctx.strokeStyle = 'white';
                        } else {
                            ctx.strokeStyle = '#444444';
                        }
                    }
                    ctx.lineWidth = 3;
                    ctx.rect(bout.x, bout.y, bout.width, bout.height);
                    ctx.fill();
                    ctx.stroke();
                    state.behaviorCanvases[behaviorId].style.cursor = 'auto';
                }
            });
        }, false);

        if (state.unlocked) {
            state.behaviorCanvases[behaviorId].addEventListener('mousemove', (e) => {
                editBout(e, behaviorId)
            }, false);

            state.behaviorCanvases[behaviorId].tabIndex = '1';
            state.behaviorCanvases[behaviorId].addEventListener('keyup', (e) => {
                removeBout(e, behaviorId);
                expandContractBout(e, behaviorId);
                translateBout(e, behaviorId)
                toggleAutoManual(e, behaviorId);
                shiftBout(e, behaviorId);
            });

            state.behaviorCanvases[behaviorId].addEventListener('dblclick', (e) => {
                addBout(e, behaviorId);
            });

            state.behaviorCanvases[behaviorId].addEventListener('mousedown', (e) => {
                whenMouseDown();
            });

            state.behaviorCanvases[behaviorId].addEventListener('mouseup', (e) => {
                whenMouseUp();
            });
        }
    });

    if (state.unlocked) {
        document.querySelectorAll('.behaviorContainer').forEach(container => {
            changeBehaviorName(container);
        });
    }
}


function changeBehaviorName(container) {
    var behaviorId = container.childNodes[1].id;
    var name = container.childNodes[0];
    var oldName =  JSON.parse(JSON.stringify(name.value));
    var newName = ''; 

    name.addEventListener('change', (e) => {
        newName = name.value;
        if (Object.values(state.behaviorIds).includes(newName)) {
            name.value = oldName;
            newName = oldName;
            alert('this behavior already exists');
        }
        state.behaviorIds[behaviorId] = newName;
        Object.keys(state.behaviors).forEach(function(id) {
            if (state.behaviors[id].behavior_id === behaviorId) {

                var changedBout = {
                    behavior: state.behaviors[id].behavior,
                    behavior_id: state.behaviors[id].behavior_id,
                    bout_id: state.behaviors[id].bout_id,
                    end: state.behaviors[id].end,
                    filename: state.behaviors[id].filename,
                    folders: state.behaviors[id].folders,
                    manual: state.behaviors[id].manual,
                    session: state.behaviors[id].session,
                    start: state.behaviors[id].start
                };
                state.behaviors[id]['behavior'] = newName;
                state.changes = {
                    id: id,
                    session: state.session,
                    old: changedBout,
                    new: {behavior: newName}, 
                    modification: 'changed behavior'
                }
                state.behaviorChanges.push(state.changes);
                state.redo = [];
            }    
        }); 
        state.uniqueTrialBehaviors = Object.values(state.behaviorIds);
        console.log(state.behaviorIds)
        console.log(state.uniqueTrialBehaviors);
        drawActogram(); 
    });
    
    var behaviorIdList = Object.keys(state.behaviorIds);
    for (var i in behaviorIdList) {
        createBehavior(behaviorIdList[i], colors2[i%colors2.length]);
    }
}

function editBout(e, behaviorId) {

    var nFrames = state.videos[0].duration * state.fps;
    Object.keys(state.bouts[behaviorId]).forEach(function(key) {
        var bout = state.bouts[behaviorId][key];
        var rect = state.behaviorCanvases[behaviorId].getBoundingClientRect();
        var point = {x: e.clientX - rect.left, y: e.clientY - rect.top};
        var err = 5;
        state.bouts[behaviorId][key].right = (rect.width-2) * (bout.end/nFrames);
        state.bouts[behaviorId][key].left = (rect.width-2) * (bout.start/nFrames);
        if (bout.selected) {
            if (point.x >= (bout.left - err) && point.x <= (bout.left + err)) {
                state.behaviorCanvases[behaviorId].style.cursor = 'w-resize';
                state.expectResize = 0;
            } else if (point.x >= (bout.right - err) && point.x <= (bout.right + err)) {
                state.behaviorCanvases[behaviorId].style.cursor = 'e-resize';
                state.expectResize = 1;
            } else {
                state.behaviorCanvases[behaviorId].style.cursor = 'auto';
            }
        }

        if (state.isResizeDrag && bout.selected) {
            var oldx = state.bouts[behaviorId][key].x;
            var start = state.behaviors[bout.bout_id].start;
            var end = state.behaviors[bout.bout_id].end;
            var minFrames = 10;
            if (state.expectResize === 0) {
                state.modified = true;
                start = Math.floor((point.x / (rect.width - 2)) * nFrames);
                if (start >= end) {
                    start = Math.max(0, end - minFrames); 
                }
            } else if (state.expectResize === 1) {
                state.modified = true;
                end = Math.floor((point.x / (rect.width - 2)) * nFrames);
                if (end <= start){
                    end = Math.min(start + minFrames, nFrames);
                }
                // var timeToSet = (end/nFrames)*state.videos[0].duration;
                // for(var i=0; i<state.videos.length; i++) {
                //     state.videos[i].currentTime = timeToSet;
                // }
                // drawFrame(true);
            }
            state.behaviors[bout.bout_id].start = start; 
            state.behaviors[bout.bout_id].end = end;
            updateBehaviorState(behaviorId, bout.color, rect);
        }

        if (state.isDrag && bout.selected) {
            state.modified = true;
            var oldStart = state.behaviors[bout.bout_id].start;
            var start = Math.floor((point.x / (rect.width - 2)) * nFrames) - state.selectedFrameOffset;
            if (start < 0) {
                start = 0;
            }
            var end = Math.floor(start + (state.behaviors[bout.bout_id].end - oldStart))
            if (end > nFrames) {
                end = nFrames;
                start = nFrames - (state.behaviors[bout.bout_id].end - oldStart);
            }
            state.behaviors[bout.bout_id].start = start;
            state.behaviors[bout.bout_id].end = end;
            updateBehaviorState(behaviorId, bout.color, rect);
        }
    });
}


function expandContractBout(e, behaviorId) {

    if (!state.selectedBout) {
        return;
    }

    Object.keys(state.bouts[behaviorId]).forEach(function(id) {
        var bout = state.bouts[behaviorId][id];
        var nFrames = state.videos[0].duration * state.fps;
        var behaviorCanvas = state.behaviorCanvases[state.selectedBehavior];
        var rect = state.behaviorCanvases[behaviorId].getBoundingClientRect();

        state.changes = {
                id: id,
                session: state.session,
                modification: 'edited'
            };

        state.changes.old = {
            bout_id: state.behaviors[id].bout_id,
            behavior_id: state.behaviors[id].behavior_id, 
            session: state.session, 
            folders: state.behaviors[id].folders, 
            filename:state.behaviors[id].filename,
            start: state.behaviors[id].start,
            end: state.behaviors[id].end,
            behavior: state.behaviors[id].behavior, 
            manual: state.behaviors[id].manual
        };

        if (bout.selected && e.shiftKey) {
            switch(e.which) {
                case 37:
                    state.behaviors[id].start = Math.max(0, state.behaviors[id].start - 1); 
                    updateBehaviorState(behaviorId, bout.color, rect);
                    state.changes.new = {start: state.behaviors[id].start, manual: true};
                    state.behaviorChanges.push(state.changes);
                    state.redo = [];
                    break;
                case 39:
                    state.behaviors[id].start = Math.min(state.behaviors[id].start + 1, nFrames); 
                    updateBehaviorState(behaviorId, bout.color, rect);
                    state.changes.new = {start: state.behaviors[id].start, manual: true};
                    state.behaviorChanges.push(state.changes);
                    state.redo = [];
                    break;
                default:
                    break;
            }
            drawNextFrame(true, state.behaviors[id].start);
        }

        if (bout.selected && e.ctrlKey) {
            switch(e.which) {
                case 37:
                    state.behaviors[id].end = Math.max(0, state.behaviors[id].end - 1); 
                    updateBehaviorState(behaviorId, bout.color, rect);
                    state.changes.new = {end: state.behaviors[id].end, manual: true};
                    state.behaviorChanges.push(state.changes);
                    state.redo = [];
                    break;
                case 39:
                    state.behaviors[id].end = Math.min(state.behaviors[id].end + 1, nFrames); 
                    updateBehaviorState(behaviorId, bout.color, rect);
                    state.changes.new = {end: state.behaviors[id].end, manual: true};
                    state.behaviorChanges.push(state.changes);
                    state.redo = [];
                    break;
                default:
                    break;
            }
            drawNextFrame(true, state.behaviors[id].end);
        }
    });
}

function shiftBout(e, behaviorId) {

    if (!state.selectedBout) {
        return; 
    }

    Object.keys(state.bouts[behaviorId]).forEach(function(id) {
        console.log(state.bouts)
        var bout = state.bouts[behaviorId][id];
        if (bout.selected) {
            switch(e.which) {

                case 38: // up
                    var currIx = state.behaviorOrder[behaviorId];
                    if (state.behaviorOrder[behaviorId] == 0) {
                        var nextBehaviorId = state.behaviorOrder[Object.keys(state.behaviorIds).length - 1];
                    } else {
                        var nextBehaviorId = state.behaviorOrder[currIx - 1];
                    }
                    var start = state.behaviors[id].start;
                    var end = state.behaviors[id].end;
                    var newId = add(nextBehaviorId, start, end);
                    remove(behaviorId, id, redraw = true);
                    break;

                case 40: // down
                    var currIx = state.behaviorOrder[behaviorId];
                    if (state.behaviorOrder[behaviorId] == Object.keys(state.behaviorIds).length - 1) {
                        var nextBehaviorId = state.behaviorOrder[0];
                    } else {
                        var nextBehaviorId = state.behaviorOrder[currIx + 1];
                    }
                    var start = state.behaviors[id].start;
                    var end = state.behaviors[id].end;
                    var newId = add(nextBehaviorId, start, end);
                    remove(behaviorId, id, redraw = true);
                    break;

                default:
                    break;
            }
        }
    });

}


function translateBout(e, behaviorId) {

    if (!state.selectedBout) {
        return;
    }

    Object.keys(state.bouts[behaviorId]).forEach(function(id) {
        var bout = state.bouts[behaviorId][id];
        var nFrames = state.videos[0].duration * state.fps;
        var behaviorCanvas = state.behaviorCanvases[state.selectedBehavior];
        var rect = state.behaviorCanvases[behaviorId].getBoundingClientRect();

        state.changes = {
                id: id,
                session: state.session,
                modification: 'edited'
            };

        state.changes.old = {
            bout_id: state.behaviors[id].bout_id,
            behavior_id: state.behaviors[id].behavior_id, 
            session: state.session, 
            folders: state.behaviors[id].folders, 
            filename:state.behaviors[id].filename,
            start: state.behaviors[id].start,
            end: state.behaviors[id].end,
            behavior: state.behaviors[id].behavior, 
            manual: state.behaviors[id].manual
        };

        if (bout.selected && !e.ctrlKey && !e.shiftKey) {
            switch(e.which) {
                case 37:
                    if (state.behaviors[id].start == 0) {
                        break;
                    }
                    state.behaviors[id].start = state.behaviors[id].start - 1; 
                    state.behaviors[id].end = state.behaviors[id].end - 1;
                    updateBehaviorState(behaviorId, bout.color, rect);
                    state.changes.new = {start: state.behaviors[id].start, end: state.behaviors[id].end, manual: true};
                    state.behaviorChanges.push(state.changes);
                    state.redo = [];
                    break;
                case 39:
                    if (state.behaviors[id].end == nFrames) {
                        break;
                    }
                    state.behaviors[id].start = state.behaviors[id].start + 1;
                    state.behaviors[id].end = state.behaviors[id].end + 1; 
                    updateBehaviorState(behaviorId, bout.color, rect);
                    state.changes.new = {start: state.behaviors[id].start, end: state.behaviors[id].end, manual: true};
                    state.behaviorChanges.push(state.changes);
                    state.redo = [];
                    break;
                default:
                    break;
            }
            drawNextFrame(true, state.behaviors[id].start);
        }
    });
}

function toggleAutoManual(e, behaviorId) {

    if (!state.selectedBout) {
        return;
    }

    Object.keys(state.bouts[behaviorId]).forEach(function(id) {
        var bout = state.bouts[behaviorId][id];
        var rect = state.behaviorCanvases[behaviorId].getBoundingClientRect();
        if (e.key === "Enter" && bout.selected) { 
            var oldBout = JSON.parse(JSON.stringify(state.behaviors[id]));
            state.changes = {
                id: id,
                session: state.session,
                old: oldBout,
                new: {manual: !oldBout.manual}, 
                modification: 'manual'
            }
            state.changes.old.session = state.session;
            state.behaviors[id].manual = !state.behaviors[id].manual;
            state.bouts[behaviorId][id].manual = !state.behaviors[id].manual;
            state.behaviors[id].selected = false;
            state.bouts[behaviorId][id].selected = false;
            state.behaviorChanges.push(state.changes);
            state.redo = [];
            var ctx = state.behaviorCanvases[behaviorId].getContext("2d");
            drawBehavior(behaviorId, ctx); 
            drawActogram();
        }
    });

}

function drawButtons() {
    var defaultButtons = document.getElementById('defaultButtons');
    var editingButtons = document.getElementById('editingButtons');
    if (state.unlocked) {
        editingButtons.style.display = 'block';
        defaultButtons.style.display = 'none';
    } else {
        editingButtons.style.display = 'none';
        defaultButtons.style.display = 'block';
    }
}

function removeBout(e, behaviorId) {

    if (!state.selectedBout) {
        return;
    }

    Object.keys(state.bouts[behaviorId]).forEach(function(id) {
        var bout = state.bouts[behaviorId][id];
        if (e.key === 'Backspace' && bout.selected) {
            remove(behaviorId, id);
        }
    });
}

function remove(behaviorId, id, redraw = true) {
    var rect = state.behaviorCanvases[behaviorId].getBoundingClientRect();
    var removedBout = state.behaviors[id];
    state.changes = {
        id: id,
        session: state.session,
        old: removedBout,
        new: {}, 
        modification: 'removed'
    };
    state.behaviorChanges.push(state.changes);
    state.redo = [];
    delete state.behaviors[id];
    delete state.bouts[behaviorId][id];
    state.selectedBout = undefined;
    state.selectedBehavior = undefined;
    updateBehaviorState(behaviorId, state.behaviorCanvases[behaviorId].style.borderColor, rect);
    console.log(state.behaviorChanges);
    state.uniqueTrialBehaviors = Object.values(state.behaviorIds);
    if (redraw) {
        drawActogram();
    }
}


function addBout(e, behaviorId) {

    if (state.selectedBout) {
        return;
    }

    var nFrames = state.videos[0].duration * state.fps;
    var rect = state.behaviorCanvases[behaviorId].getBoundingClientRect();
    var point = {x: e.clientX - rect.left, y: e.clientY - rect.top};
    var length = Math.floor(nFrames / 30);
    var start = Math.floor((point.x / (rect.width - 2)) * nFrames);
    var end = Math.min(start + length, nFrames);

    add(behaviorId, start, end);
    
}

function add(behaviorId, start, end) {

    var ctx = state.behaviorCanvases[behaviorId].getContext("2d");
    var newId = generateId(22);
    
    var addedBout = {
        session: state.trial.session,
        filename: state.trial.vidname,
        folders: state.trial.folder,
        start: start,
        end: end,
        bout_id: newId,
        behavior: state.behaviorIds[behaviorId],
        behavior_id: behaviorId,
        manual: true
    };

    state.behaviors[newId] = addedBout;
    state.selectedBehavior = behaviorId;
    state.selectedBout = addedBout.bout_id;
    state.uniqueTrialBehaviors = Object.values(state.behaviorIds);
    var behaviorIdList = Object.keys(state.behaviorIds);
    for (var i in behaviorIdList) {
        createBehavior(behaviorIdList[i], colors2[i%colors2.length]);
    }
    selectBout(ctx);

    state.changes = {
        id: newId,
        session: state.session, 
        old: {},
        new: addedBout, 
        modification: 'added'
    }
    state.behaviorChanges.push(state.changes);
    state.redo = [];

    return newId;
}


function whenMouseDown() {

    state.changes = {};
    if (state.expectResize !== -1) {
        state.isResizeDrag = true;
    } else if (state.selectedBout) {
        state.isDrag = true;
    }

    if (state.isResizeDrag || state.isDrag) {
        var currentBout = state.behaviors[state.selectedBout];
        state.changes.id = currentBout.bout_id;
        state.changes.session = state.session;
        state.changes.old = JSON.parse(JSON.stringify(state.behaviors[state.selectedBout]));
        state.changes.old.session = state.session;
        // state.changes.old = {
        //     bout_id: currentBout.bout_id,
        //     behavior_id: currentBout.behavior_id, 
        //     session: state.session, 
        //     folders: currentBout.folders, 
        //     filename:currentBout.filename,
        //     start: currentBout.start,
        //     end: currentBout.end,
        //     behavior: currentBout.behavior, 
        //     manual: currentBout.manual
        // };
    }
}

function whenMouseUp() {

    if (state.modified && (state.isResizeDrag || state.isDrag)) {
        var newBout = state.behaviors[state.changes.id];
        state.changes.modification = 'edited';
        state.changes.new = {start: newBout.start, end: newBout.end, manual: true};
        state.behaviorChanges.push(state.changes);
        state.redo = [];
        state.modified = false;
    }

    state.expectResize = -1;
    state.isResizeDrag = false;
    state.isDrag = false;
}

function selectBout(ctx) {

    var bout = state.bouts[state.selectedBehavior][state.selectedBout];
    var nFrames = state.videos[0].duration * state.fps;
    var behaviorCanvas = state.behaviorCanvases[state.selectedBehavior]

    var timeToSet = (bout.start/nFrames)*state.videos[0].duration;
    for(var i=0; i<state.videos.length; i++) {
        state.videos[i].currentTime = timeToSet;
    }
    drawFrame(true);

    state.bouts[state.selectedBehavior][state.selectedBout].selected = true;
    ctx.beginPath();
    ctx.fillStyle = 'white';
    ctx.lineWidth = 2;
    ctx.rect(bout.x, bout.y, bout.width, bout.height);
    ctx.fill();
    if (state.unlocked) {
        ctx.strokeStyle = bout.color;
        ctx.stroke();
    }
}

function isSelected(point, bout) {
    return (point.x > bout.left && point.x < bout.right);
}

function updateBehaviorState(behaviorId, color, rect) {

    if (state.selectedBout) {
        var nFrames = state.videos[0].duration * state.fps;
        var id = state.selectedBout;
        state.behaviors[id].manual = true;
        console.log(state.selectedBout);
        state.bouts[behaviorId][state.selectedBout] = {
            bout_id: state.behaviors[id]['bout_id'],
            start: state.behaviors[id]['start'],
            end: state.behaviors[id]['end'],
            x: state.behaviorCanvases[behaviorId].width*(state.behaviors[id]['start']/nFrames),
            y: 0,
            width: state.behaviorCanvases[behaviorId].width*((state.behaviors[id]['end']-state.behaviors[id]['start'])/nFrames),
            height: state.behaviorCanvases[behaviorId].height,
            right: (rect.width-2) * (state.behaviors[id]['end']/nFrames),
            left: (rect.width-2) * (state.behaviors[id]['start']/nFrames),
            color: color, 
            selected: true,
            manual: state.behaviors[id].manual
        };

    }; 

    var ctx = state.behaviorCanvases[behaviorId].getContext("2d");
    state.behaviorCanvases[behaviorId].style.border ='1px solid ' + color;
    ctx.clearRect(0, 0, state.behaviorCanvases[behaviorId].width, state.behaviorCanvases[behaviorId].height);      
    drawBehavior(behaviorId, ctx);
}


function getBoutColor(ctx, bout, behaviorId) {
    var color = undefined;
    if (bout.selected) {
        color = 'white';
    } else if (state.behaviors[bout.bout_id].manual) {
        color = state.behaviorCanvases[behaviorId].style.borderColor;
    } else {
        // color = 'gray'
        var pc = document.createElement('canvas');
        // pc.width = state.behaviorCanvases[behaviorId].width/30;
        pc.width = 20;
        pc.height = state.behaviorCanvases[behaviorId].height;
        var pctx = pc.getContext('2d');
        pctx.fillStyle = state.behaviorCanvases[behaviorId].style.borderColor;
        pctx.fillRect(0, 0, pc.width, pc.height);
        // pctx.lineWidth = pc.width / 3;
        pctx.lineWidth = 4;
        pctx.strokeStyle = '#444444';
        ctx.beginPath();
        pctx.moveTo(2, 0);
        pctx.lineTo(pc.width-10, pc.height);
        pctx.stroke();
        ctx.closePath();
        color = ctx.createPattern(pc, 'repeat-x');
        // ctx.fillStyle = color;
    }
    return ctx, color;
}

function drawBehavior(behaviorId, ctx) {

    Object.keys(state.bouts[behaviorId]).forEach(function(key) {
        var bout = state.bouts[behaviorId][key];
        ctx, color = getBoutColor(ctx, bout, behaviorId);
        ctx.beginPath();
        ctx.fillStyle = color;
        if (state.unlocked) {
            if(state.behaviors[bout.bout_id].manual) {
                ctx.strokeStyle = 'white';
            } else {
                ctx.strokeStyle = '#444444';
            }
        }
        ctx.lineWidth = 3;
        ctx.rect(bout.x, bout.y, bout.width, bout.height);
        ctx.fill();
        ctx.stroke();
    });
}

function createBehavior(behaviorId, color) {

    var behavior = state.behaviorIds[behaviorId];
    var nFrames = state.videos[0].duration * state.fps;
    var ctx = state.behaviorCanvases[behaviorId].getContext("2d");
    state.behaviorCanvases[behaviorId], ctx = updateCanvas(state.behaviorCanvases[behaviorId], ctx);
    state.behaviorCanvases[behaviorId].style.border ='1px solid ' + color;

    var bouts = {};
    Object.keys(state.behaviors).forEach(function(id) {
        if (state.behaviors[id]['behavior'] == state.behaviorIds[behaviorId]) { 
            bout = {
                bout_id: state.behaviors[id]['bout_id'],
                start: state.behaviors[id]['start'],
                end: state.behaviors[id]['end'],
                x: state.behaviorCanvases[behaviorId].width*(state.behaviors[id]['start']/nFrames),
                y: 0,
                width: state.behaviorCanvases[behaviorId].width*((state.behaviors[id]['end']-state.behaviors[id]['start'])/nFrames),
                height: state.behaviorCanvases[behaviorId].height,
                color: color, 
                selected: false
            };
            bouts[bout.bout_id] = bout;
        }
    });
    state.bouts[behaviorId] = bouts;
    drawBehavior(behaviorId, ctx); 
}

function updateCanvas(behaviorCanvas, ctx) {

    ctx.translate(0.5, 0.5);
    var sizeWidth = 80 * window.innerWidth / 100;
    var sizeHeight = 100 * window.innerHeight / 100 || 766; 

    behaviorCanvas.width = sizeWidth;
    behaviorCanvas.height = sizeHeight;
    behaviorCanvas.style.width = sizeWidth;
    behaviorCanvas.style.height = sizeHeight;

    return behaviorCanvas, ctx
}

function generateId(length) {
    var id = '';
    var characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_-';
    for (var i = 0; i < length; i++ ) {
        id += characters.charAt(Math.floor(Math.random() * characters.length));
    }
   return id;
}

function updateProgressBar() {
    var video = state.videos[0];
    var progressBar = document.getElementById('progressBar');
    if(video && video.duration && video.currentTime) {
        // var percentage = Math.floor((100 / video.duration) * video.currentTime);
        var value = (100 / video.duration) * video.currentTime;
        var percentage = Math.round((value + Number.EPSILON) * 1000) / 1000;
        progressBar.value = percentage;
        progressBar.innerHTML = percentage + '% played';
    } else {
        progressBar.value = 0;
        progressBar.innerHTML = "";
    }
}

function updateFrameNumber() {
    var video = state.videos[0];
    var currTime = video.currentTime;
    var currFrame = Math.floor(currTime*state.fps);
    return currFrame
}

// Set the play position of the video based on the mouse click at x
function setPlayPosition(x) {
    var progressBar = document.getElementById("progressBar");
    var value = (x - findPos(progressBar));
    var timeToSet = ((state.videos[0].duration / progressBar.offsetWidth) * value);

    for(var i=0; i<state.videos.length; i++) {
        state.videos[i].currentTime = timeToSet;
    }
    drawFrame(true);
}

// Find the real position of obj
function findPos(obj) {
    var curleft = 0;
    if (obj.offsetParent) {
        do { curleft += obj.offsetLeft; } while (obj = obj.offsetParent);
    }
    return curleft;
}

function play() {
    playing = true;
    var t = state.videos[0].currentTime;
    var nFrames = state.videos[0].duration * state.fps;
    framenum = state.videos[0].currentTime * state.fps;
    // rate_estimate = state.vid_fps/state.fps*slowdown;
    for(var i=0; i<state.videos.length; i++) {        
        state.videos[i].currentTime = t;
        state.videos[i].playbackRate = slowdown;
        state.videos[i].loop = true;
        state.videos[i].preload = "auto";
        state.videos[i].play();
    }

    // state.videos[0].play();
    setTimeout(drawFrame, 150.0);
}

function pause() {
    var t = state.videos[0].currentTime;
    for(var i=0; i<state.videos.length; i++) {
        state.videos[i].currentTime = t;
        state.videos[i].pause();
    }
    t = state.videos[0].currentTime;
    var framenum = Math.round(t * state.fps);
    playing = false;
    if(state.data) {
        updateKeypoints(state.data[framenum])
        draw2D(framenum);
    }
}

function clearUnusedBehavior() {
    state.uniqueTrialBehaviors = getUniqueTrialBehaviors();
    drawActogram();
}

function addBehavior() {
    state.uniqueTrialBehaviors = Object.values(state.behaviorIds);
    var behaviorId = generateId(10);
    state.uniqueTrialBehaviors.push(behaviorId);
    state.behaviorIds[behaviorId] = behaviorId;
    drawActogram();

    var name = 'name' + (state.uniqueTrialBehaviors.length-1).toString();
    document.getElementById(name).focus();
}

function unlockEditing() {
    var password = prompt("password:");

    fetch('/unlock-editing', {
        headers: {'Content-Type': 'application/json'},
        method: 'POST',
        body: JSON.stringify({password})

    }).then(function (response) {
        return response.json();

    }).then(function (res) {
        state.token = res['token'];
        if (res['valid'] && state.token != -1) {
            console.log('unlocked')
            setCookie('token', res['token'])
            unlock();
        }       
    });
    console.log(state.unlocked)
    console.log(getCookie('token'))
}

function unlock() {
    state.unlocked = true;
    state.modal = document.getElementById('keyboardShortcuts');
    drawButtons();
    drawActogram();
}

function getCookie(name) {
    return localStorage.getItem(name);
}

function setCookie(name, value) {
    return localStorage.setItem(name, value);
}

function pushChanges() {

    var video = state.trial.session + "/" + state.trial.folder + "/" + state.trial.vidname;
    for (var i=0; i<state.behaviorChanges.length; i++) {
        if (!state.allBehaviorChanges[video]) {
            state.allBehaviorChanges[video] = [];
        } 
        state.allBehaviorChanges[video].push(state.behaviorChanges[i])
    }
    var allBehaviorChanges = state.allBehaviorChanges;
    var token = state.token;

    fetch('/update-behavior', {
        headers: {'Content-Type': 'application/json'},
        method: 'POST',
        body: JSON.stringify({allBehaviorChanges, token})

    }).then(function (response) { 
        return response.text();

    }).then(function (text) {
        alert(text);
    });

    // updateTrial(state.trial);
    for (var i=0; i<state.behaviorChanges.length; i++) {
        if (!state.allBehaviorChanges[video]) {
            state.allBehaviorChanges[video] = [];
        } 
        state.allBehaviorChanges[video].push(state.behaviorChanges[i])
    }
    state.behaviorChanges = [];
    state.redo = [];
    state.allBehaviorChanges = {};
    updateTrial(state.trial)
}

function togglePlayPause() {
    if(!playing) {
        play();
    } else {
        pause();
    }
    updatePlayPauseButton();
}

function updatePlayPauseButton() {
    var button = document.getElementById("play");
    if(playing) {
        button.innerHTML = "pause";
    } else {
        button.innerHTML = "play";
    }
}

function toggle2D() {
    if (!display2d) {
        display2d = true;
    } else {
        display2d = false;
    }
    draw2D(framenum);
    updateToggle2DButton();
}

function updateToggle2DButton() {
    var button = document.getElementById("toggle2d");
    if(display2d) {
        button.innerHTML = "hide 2d";
    } else {
        button.innerHTML = "display 2d";
    }
}


function toggle3D() {
    var container = document.getElementById("modelContainer");
    if (!display3d) {
        display3d = true;
        container.classList.remove("hidden");
    } else {
        display3d = false;
        container.classList.add("hidden");
    }
    updateToggle3DButton();
}

function updateToggle3DButton() {
    var button = document.getElementById("toggle3d");
    if(display3d) {
        button.innerHTML = "hide 3d";
    } else {
        button.innerHTML = "display 3d";
    }
}

function slowdownVideo() {
    slowdown = slowdown / Math.sqrt(2);
    if(playing) { play(); }
    updateSpeedText();
}

function speedupVideo() {
    slowdown = slowdown * Math.sqrt(2);
    if(playing) { play(); }
    updateSpeedText();
}

function advanceFrame(num) {
    framenum += num;
    drawNextFrame(true, framenum);
}

function updateSpeedText() {

    if (!state.metadata) {
        return;
    }

    var full_slow = slowdown * state.metadata.video_speed;
    var text = "";
    if(Math.abs(full_slow - 1.0) < 1e-3) {
        text = "actual speed";
    } else if(full_slow < 1.0) {
        text = "slowed x" + (1/full_slow).toFixed(1);
    } else if(full_slow > 1.0) {
        text = "sped up x" + full_slow.toFixed(1);
    }
    var span = document.getElementById("speed");
    span.innerHTML = text;
}

function updateKeypoints(kps) {

    if (!kps || !state.metadata) {
        return;
    }

    var scale = 3;
    for(var i=0; i<kps.length; i++) {
        var kp = kps[i];
        var is_bad = (kp[0] == 0 && kp[1] == 0 && kp[2] == 0);
        if(is_bad) {
            kp[0] = NaN;
            kp[1] = NaN;
            kp[2] = NaN;
        }
        if (!state.spheres) {
            drawSpheres(state.scene, kps, scale);
            drawTubes(state.scene, state.metadata.scheme, kps, scale);
        } else {
            state.spheres[i].position.x = kp[0]*scale;
            state.spheres[i].position.y = kp[1]*scale;
            state.spheres[i].position.z = -kp[2]*scale;
        }
    }

    var tubecount = 0;
    for(var i=0; i<state.metadata.scheme.length; i++) {
        var links = state.metadata.scheme[i];
        var prev = null;
        for(var j=1; j<links.length; j++) {
            var prev = kps[links[j-1]];
            var vec = kps[links[j]];
            // var vec = new BABYLON.Vector3(kp[0]*scale, kp[1]*scale, -kp[2]*scale);
            state.paths[tubecount][0].x = prev[0]*scale;
            state.paths[tubecount][0].y = prev[1]*scale;
            state.paths[tubecount][0].z = -prev[2]*scale;
            state.paths[tubecount][1].x = vec[0]*scale;
            state.paths[tubecount][1].y = vec[1]*scale;
            state.paths[tubecount][1].z = -vec[2]*scale;

            var tube = BABYLON.MeshBuilder.CreateTube(
                null,
                {path: state.paths[tubecount],
                 instance: state.tubes[tubecount]});
            tubecount++;
        }
    }
}

function drawSpheres(scene, keypoints, scale) { 

    if (!keypoints) {
        return;
    }

    state.spheres = [];

    for(var i=0; i<keypoints.length; i++) {
        var kp = keypoints[i];
        // This is where you create and manipulate meshes
        var sphere = BABYLON.MeshBuilder.CreateSphere(
            "sphere",
            { diameter: 0.25, updatable: true },
            scene
        );
        sphere.position = new BABYLON.Vector3(kp[0]*scale, kp[1]*scale, -kp[2]*scale);

        var mat = new BABYLON.StandardMaterial("material", scene);
        mat.ambientColor = new BABYLON.Color3(1, 1, 1);
        sphere.material = mat;

        state.spheres.push(sphere);
    }
}

function drawTubes(scene, scheme, keypoints, scale) {

    if (!keypoints) {
        return;
    }

    state.tubes = [];
    state.paths = [];
    for(var i=0; i<scheme.length; i++) {
        var links = scheme[i];
        var prev = null;
        var col = colors[i];
        for(var j=0; j<links.length; j++) {
            var kp = keypoints[links[j]]
            var vec = new BABYLON.Vector3(kp[0]*scale, kp[1]*scale, -kp[2]*scale);
            if(j != 0) {
                var path = [prev, vec];
                // draw limbs
                var tube = BABYLON.MeshBuilder.CreateTube(
                    "tube",
                    {path: path, radius: 0.05,
                     sideOrientation: BABYLON.Mesh.DOUBLESIDE,
                     cap: BABYLON.Mesh.CAP_ALL,
                     updatable: true},
                    scene);

                var mat = new BABYLON.StandardMaterial("material", scene);
                // mat.ambientColor = new BABYLON.Color3(col[0], col[1], col[2]);
                mat.ambientColor = new BABYLON.Color3.FromHexString(col);
                tube.material = mat;

                state.tubes.push(tube);
                state.paths.push(path);
            }
            prev = vec;
        }
    }
}

function drawPath(ctx, path, color) {
    if(!display2d) return; 

    ctx.beginPath();
    ctx.lineWidth = 2;
    ctx.strokeStyle = color;
    for(var i=0; i<path.length; i++) {
        var pt = path[i];
        if(i == 0) {
            ctx.moveTo(pt[0], pt[1]);
        } else {
            ctx.lineTo(pt[0], pt[1]);
        }
    }
    ctx.stroke();
}

function drawPoint(ctx, x, y, color) {
    if(!display2d) return; 

    ctx.beginPath();
    ctx.arc(x, y, 2, 0, 2 * Math.PI, false);
    ctx.fillStyle = color;
    ctx.fill();
//     ctx.strokeStyle = "black";
//     ctx.stroke();
}

function draw2D(framenum) {
    if(!state.data2d) return;

    for(var vidnum=0; vidnum<state.videos.length; vidnum++) {
        var vid = state.videos[vidnum];
        var ratio = vid.clientWidth / vid.videoWidth;
        var canvas = state.canvases[vidnum];
        var ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        var cname = state.camnames[vidnum];
        var kps = state.data2d[cname][framenum];
        for(var i=0; i<state.metadata.scheme.length; i++) {
            var links = state.metadata.scheme[i];
            var col = colors[i];
            var path = [];
            for(var j=0; j<links.length; j++) {
                var kp = kps[links[j]]
                if(kp[0] == 0 && kp[1] == 0) continue; // missing data
                path.push([kp[0]*ratio, kp[1]*ratio]);
            }
            drawPath(ctx, path, col);
        }
        for(var i=0; i<kps.length; i++) {
            var kp = kps[i];
            if(kp[0] == 0 && kp[1] == 0) continue; // missing data
            drawPoint(ctx, kp[0]*ratio, kp[1]*ratio, "white");
        }
    }
}
