import {SatPointCloud} from './sat_point_cloud';
import {Sat3d} from './sat3d';
import {SatLabel} from '../sat';
import {Box3d} from './box3d';

/**
 *
 * @constructor
 */
export function Sat3dTracker() {
    Sat3d.call(this, SatPointCloud, Box3d);
    if (this.tracks == null) {
        this.tracks = [];
    }
}

Sat3dTracker.prototype = Object.create(Sat3d.prototype);

Sat3dTracker.prototype.newLabel = function(optionalAttributes=null) {
    let labelId = this.newLabelId();
    let track = new Box3dTrack(this, labelId, optionalAttributes);

    this.labelIdMap[track.id] = track;
    this.labels.push(track);
    this.tracks.push(track);

    let currentLabel = null;

    for (let i = this.currentItem.index; i < this.items.length; i++) {
        let labelId = this.newLabelId();
        let childLabel = new this.LabelType(this, labelId, optionalAttributes);

        this.labelIdMap[childLabel.id] = childLabel;
        this.labels.push(childLabel);

        childLabel.parent = track;
        track.addChild(childLabel);

        this.items[i].labels.push(childLabel);
        if (i != this.currentItem.index) {
            this.items[i].addBoundingBox(childLabel,
                this.currentItem.target, false, false);
        } else {
            currentLabel = childLabel;
            childLabel.keyframe = true;
        }
    }
    return currentLabel;
};

Sat3dTracker.prototype.toJson = function() {
    let json = this.encodeBaseJson();
    json.tracks = [];
    for (let i = 0; i < this.tracks.length; i++) {
        if (this.tracks[i].valid) {
            json.tracks.push(this.tracks[i].toJson());
        }
    }
    json.task.projectOptions.metadata = this.metadata;
    return json;
};

Sat3dTracker.prototype.fromJson = function(json) {
    this.decodeBaseJson(json);
    this.metadata = json.task.projectOptions.metadata;
    this.tracks = [];
    for (let i = 0; json.tracks && i < json.tracks.length; i++) {
        let track = new Box3dTrack(this, json.tracks[i].id,
            json.tracks[i].attributes);
        track.children = [];
        for (let j = 0; j < json.tracks[i].childrenIds.length; j++) {
            let child = this.labelIdMap[json.tracks[i].childrenIds[j]];
            track.addChild(child);
            child.parent = track;
            child.setColor(child.color());
            for (let k = 0; k < this.labels.length; k++) {
                if (this.labels[k].id === track.id) {
                    this.labels[k] = track;
                }
            }
        }
        this.labelIdMap[json.tracks[i].id] = track;
        this.tracks.push(track);
    }
};

Sat3dTracker.prototype.gotoItem = function(index) {
    let self = this;
    self.broadcastViews(self.currentItem);
    Sat3d.prototype.gotoItem.call(this, index);
};

Sat3dTracker.prototype.moveSlider = function() {
    let self = this;
    self.broadcastViews(self.currentItem);
    Sat3d.prototype.moveSlider.call(this);
};

Sat3dTracker.prototype.broadcastViews = function(item) {
    for (let i = 0; i<this.items.length; i++) {
        this.items[i].target.copy(item.target);
        this.items[i].sphere.copy(item.sphere);
        for (let j = 0; j<item.views.length; j++) {
            this.items[i].views[j].camera.position.copy(
                item.views[j].camera.position);
            this.items[i].views[j].camera.up.copy(item.views[j].camera.up);
            this.items[i].views[j].camera.lookAt(this.items[i].target);
        }
    }
};

/**
 *
 * @param {Sat} sat
 * @param {int} id
 * @param {object} optionalAttributes
 * @constructor
 */
function Box3dTrack(sat, id, optionalAttributes=null) {
    SatLabel.call(this, sat, id, optionalAttributes);
}

Box3dTrack.prototype = Object.create(SatLabel.prototype);

Box3dTrack.prototype.endTrack = function(endLabel) {
    let endIndex = null;
    for (let i = 0; i < this.children.length; i++) {
        if (this.children[i].id === endLabel.id) {
            endIndex = i;
            break;
        }
    }

    if (!endLabel) {
        return;
    }

    for (let i = endIndex + 1; i < this.children.length; i++) {
        this.children[i].delete();
    }

    this.children = this.children.slice(0, endIndex + 1);

    for (let i = endIndex+ 1; i < this.sat.items.length; i++) {
        this.sat.items[i].deleteInvalidLabels();
    }
};

Box3dTrack.prototype.interpolate = function(newKeyframeLabel) {
    let newKeyframeIndex = null;

    for (let i = 0; i < this.children.length; i++) {
        if (this.children[i].id == newKeyframeLabel.id) {
            newKeyframeIndex = i;
            break;
        }
    }

    if (newKeyframeIndex == null) {
        return;
    }

    this.children[newKeyframeIndex].keyframe = true;

    // Find previous keyframe
    let prevKeyframeIndex = newKeyframeIndex - 1;
    while (prevKeyframeIndex >= 0) {
        if (this.children[prevKeyframeIndex].keyframe) {
            break;
        }
        prevKeyframeIndex--;
    }

    // Find next keyframe
    let nextKeyframeIndex = newKeyframeIndex + 1;
    while (nextKeyframeIndex < this.children.length) {
        if (this.children[nextKeyframeIndex].keyframe) {
            break;
        }
        nextKeyframeIndex++;
    }

    // Call interpolate on all labels that should be interpolated
    if (prevKeyframeIndex >= 0) {
        for (let i = prevKeyframeIndex + 1; i < newKeyframeIndex; i++) {
            this.children[i].interpolate(prevKeyframeIndex, newKeyframeIndex,
                                         i, newKeyframeLabel.attributes);
        }
    }
    if (nextKeyframeIndex < this.children.length) {
        for (let i = newKeyframeIndex + 1; i < nextKeyframeIndex; i++) {
            this.children[i].interpolate(newKeyframeIndex, nextKeyframeIndex,
                                         i, newKeyframeLabel.attributes);
        }
    }
};
