DESIGN DECISIONS
================

Utilities
---------
* We are ultimately going to move away from utils.js
* Keep utils.js for now, until it is fully phased out, as many pages use it

Load
----
We want this to be an item-specific function as the behavior may change greatly (e.g. video)

IP Retrieval
------------
* We want to retrieve the client's IP
* geoip gives us more requests and more info, so we'll use it
* we'll keep ALL the info from geoip, not just the ip

Class Hierarchy
---------------
* inheritance all stems from base classes in sat.js
* no hierarchy necessary in backend
* reuse BoxImage and BoxLabel in the video task, since each image is video agnostic

Display
-------
* Use bootstrap trick for canvas resizing
* currently canvas resizing behaves badly on large screens (e.g. 2560x1440)

Miscellaneous
-------------
* currentItem is an item, not an index