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

// A detection differs from a label in that it knows its frame and category
// but nothing else (no attributes or correspondences)
type Detection struct {
	Id           int                    `json:"id" yaml:"id"`
	Frame        int                    `json:"frame" yaml:"frame"`
	CategoryPath string                 `json:"categoryPath" yaml:"categoryPath"`
	Data         map[string]interface{} `json:"data" yaml:"data"`
}
