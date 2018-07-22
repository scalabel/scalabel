package main

// Metadata describing a video
type VideoMetaData struct {
	Bitrate    string `json:"bitrate"`
	TBR        string `json:"tbr"`
	FPS        string `json:"fps"`
	TBN        string `json:"tbn"`
	TBC        string `json:"tbc"`
	NumFrames  string `json:"numFrames"`
	Resolution string `json:"resolution"`
}
