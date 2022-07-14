from turtle import left
from kivymd.app import MDApp
from kivy.uix.widget import Widget
from kivy.lang import Builder 
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.gridlayout import GridLayout
from kivymd.uix.button import MDRaisedButton
from kivy.uix.image import Image
from kivymd.uix.label import MDLabel
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees
import math
from scipy.spatial import distance as dist


class Button_Widget(MDBoxLayout): 
  
    def __init__(self, **kwargs): 
        super().__init__(**kwargs) 
        
        self.postura = 0
        self.contador = 0
        self.estado = "Incorrecto"

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.load_video, 1.0/30.0) 
  
    def load_video(self, *args):
       
       with self.mp_pose.Pose(
           static_image_mode = True ) as pose:
           

            ret, frame = self.capture.read()
            #frame initialize
            self.image_frame = frame
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            

            if results.pose_landmarks is not None:
                
                self.mp_drawing.draw_landmarks(frame, results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(128,0,250),thickness=2,circle_radius=3),
                self.mp_drawing.DrawingSpec(color=(255,255,255),thickness=2)) #blanco
            
            frame = cv2.flip(frame,1)
            buffer = cv2.flip(frame, 0).tostring()
            texture = Texture.create(size=(frame.shape[1],frame.shape[0]),colorfmt='bgr')
            texture.blit_buffer(buffer,colorfmt='bgr',bufferfmt='ubyte')
            self.ids.image.texture = texture

    def empezar_leccion1(self, *args):
        Clock.unschedule(self.load_video)
        self.postura = 1
        Clock.schedule_interval(self.leccion1, 1.0/30.0)

    def terminar_leccion1(self,*args):
        Clock.unschedule(self.leccion1)
        Clock.schedule_interval(self.load_video, 1.0/30.0)

    def leccion1(self,*args):
        print(self.contador) 
         
        with self.mp_pose.Pose(
           static_image_mode = True ) as pose:
           
            ret, frame = self.capture.read()
            #frame initialize
            self.image_frame = frame
            height, width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            
            if results.pose_landmarks is not None:
                left_thumb_x = int(results.pose_landmarks.landmark[21].x *width)
                right_thumb_x = int(results.pose_landmarks.landmark[22].y *width)
                right_thumb_y = int(results.pose_landmarks.landmark[22].y *height)

                left_elbow_x = int(results.pose_landmarks.landmark[13].x * width)
                left_elbow_y = int(results.pose_landmarks.landmark[13].y * height)

                left_shoulder_x = int(results.pose_landmarks.landmark[11].x * width)
                left_shoulder_y = int(results.pose_landmarks.landmark[11].y * height)

                right_elbow_y = int(results.pose_landmarks.landmark[14].y * height)
                right_elbow_x = int(results.pose_landmarks.landmark[14].x * width)

                right_shoulder_x = int(results.pose_landmarks.landmark[12].x * width)
                right_shoulder_y = int(results.pose_landmarks.landmark[12].y * height)

                right_heel_x = int(results.pose_landmarks.landmark[30].x * width)
                right_heel_y = int(results.pose_landmarks.landmark[30].y * height)

                left_heel_x = int(results.pose_landmarks.landmark[29].x * width)
                left_heel_y = int(results.pose_landmarks.landmark[29].y * height)

                right_foot_index_x = int(results.pose_landmarks.landmark[32].x * width)
                right_foot_index_y = int(results.pose_landmarks.landmark[32].y * height)

                left_foot_index_x = int(results.pose_landmarks.landmark[31].x * width)
                left_foot_index_y = int(results.pose_landmarks.landmark[31].y * height)
                
                right_index_x = int(results.pose_landmarks.landmark[20].x * width)
                right_index_y = int(results.pose_landmarks.landmark[20].y * height)
                
                left_index_x = int(results.pose_landmarks.landmark[19].x * width)
                left_index_y = int(results.pose_landmarks.landmark[19].y * height)

                left_eye_x = int(results.pose_landmarks.landmark[2].x * width)
                right_eye_x = int(results.pose_landmarks.landmark[5].x * width)

                right_wrist = results.pose_landmarks.landmark[16]
                left_wrist = results.pose_landmarks.landmark[15]

                right_elbow = results.pose_landmarks.landmark[14]
                left_elbow = results.pose_landmarks.landmark[13]

                right_knee = results.pose_landmarks.landmark[26]
                left_knee = results.pose_landmarks.landmark[25]

                right_foot_index = results.pose_landmarks.landmark[32]
                left_foot_index = results.pose_landmarks.landmark[31]

                right_ankle = results.pose_landmarks.landmark[28]

                right_heel = results.pose_landmarks.landmark[30]
                left_heel = results.pose_landmarks.landmark[29]
                right_thumb = results.pose_landmarks.landmark[22]

                right_eye_outer = results.pose_landmarks.landmark[3]

                left_shoulder = results.pose_landmarks.landmark[11]

                right_hip = results.pose_landmarks.landmark[24]
                
                if(self.postura == 1):
                    #distancia_wrist =  math.sqrt((right_wrist.x-left_wrist.x)**2 + (right_wrist.y-left_wrist.y)**2 + (right_wrist.z-left_wrist.z)**2)
                    self.ids.foto.source = 'imagenes/postura1.png'
                    p1 = np.array([right_wrist.x,right_wrist.y,right_wrist.z])
                    p2 = np.array([left_wrist.x,left_wrist.y,left_wrist.z])
                    distancia_wrist = np.linalg.norm( p1 - p2)
                    if(distancia_wrist<0.13 and right_wrist.y + 0.05 < right_elbow.y and right_heel.y < left_knee.y):
                        self.estado = "correcto"
                        self.ids.toolbar.title = "Empecemos la primera"
                        self.ids.foto.text = '[size=26]'+self.estado+'[/size]\n'
                        if(self.contador <= 10):
                            self.contador = self.contador + 1
                        else:
                            self.postura = 2
                            self.estado = "Incorrecto"
                    else:
                        self.contador = 0
                        self.ids.foto.text = '[size=26]'+''+'[/size]\n'
                        
                if(self.postura == 2):
                    self.ids.foto.source = 'imagenes/postura2.png'
                    self.ids.toolbar.title = "Empecemos la segunda"
                    
                    distancia_rightWrist_rightAnkle = math.sqrt((right_ankle.x-right_wrist.x)**2 + (right_ankle.y-right_wrist.y)**2 + (right_ankle.z-right_wrist.z)**2)

                    if(right_thumb.y > (right_ankle.y - 0.2) and left_wrist.y < right_eye_outer.y and left_wrist.x < left_shoulder.x):
                        self.estado = "correcto"
                        self.ids.toolbar.title = ""
                        self.ids.foto.text = '[size=26]'+self.estado+'[/size]\n'
                        if(self.contador <= 10):
                            self.contador = self.contador + 1
                        else:
                            self.postura = 3
                            self.estado = "Incorrecto"
                    else:
                        self.contador = 0
                        self.ids.foto.text = '[size=26]'+''+'[/size]\n'
                
                if (self.postura == 3):
                    self.ids.foto.source = 'imagenes/postura3.png'
                    self.ids.toolbar.title = "Empecemos la tercera"
                    
                    
                    p1 = np.array([right_hip.x,right_hip.y,right_hip.z])
                    p2 = np.array([right_knee.x,right_knee.y,right_knee.z])
                    p3 = np.array([right_ankle.x,right_ankle.y,right_ankle.z])
                    l1 = np.linalg.norm(p2 - p3)
                    l2 = np.linalg.norm(p1 - p3)
                    l3 = np.linalg.norm(p1 - p2)

                    angle = degrees(acos((l1**2 + l3**2 - l2**2)/(2*l1*l3)))
                    
                    if(angle < 140 and left_elbow.y < right_eye_outer.y): # angle < 99
                        self.estado = "correcto"
                        self.ids.toolbar.title = ""
                        self.ids.foto.text = '[size=26]'+self.estado+'[/size]\n'
                        if(self.contador <= 10):
                            self.contador = self.contador + 1
                        else:
                            self.postura = 4
                            self.estado = "Incorrecto"
                    else:
                        self.contador = 0
                        self.ids.foto.text = '[size=26]'+''+'[/size]\n'
                
                if (self.postura == 4):
                    self.ids.foto.source = 'imagenes/logo.png'
                    self.ids.foto.text = '[size=26]'+''+'[/size]\n'
                    self.ids.toolbar.title = "Hemos terminado la lección 1"
                    self.terminar_leccion1()

                self.mp_drawing.draw_landmarks(frame, results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(128,0,250),thickness=2,circle_radius=3),
                self.mp_drawing.DrawingSpec(color=(255,255,255),thickness=2)) 

            frame = cv2.flip(frame,1)
            buffer = cv2.flip(frame, 0).tostring()
            texture = Texture.create(size=(frame.shape[1],frame.shape[0]),colorfmt='bgr')
            texture.blit_buffer(buffer,colorfmt='bgr',bufferfmt='ubyte')
            self.ids.image.texture = texture     
    
  
class ButtonApp(MDApp): 
  
    def build(self): 
        self.title = "YOGA"
        self.theme_cls.primary_palette = "Blue"
        return Builder.load_file("Button_Widget.kv") 
  
if __name__ == "__main__": 
    ButtonApp().run()