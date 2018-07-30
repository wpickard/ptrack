# Intel RealSense D435 Person Tracking with 2-Axis Camera Mount

## Introduction

The goal of this project is to create a camera rig that can pivot on 2 axes and track a person as they move around a room using an Intel D435 RGB-D camera. To accomplish this task, a number of machine vision algorithms and libraries, each with their own strengths and intended uses, were evaluated to find which would best meet the project requirements. The tracking data from the machine vision algorithm was then used to control a simple 2-axis camera mount powered by two servos. An off-the-shelf Pololu Maestro servo controller is used as the interface between the computer and the servos because of its configurability and reliability.
