# DATA SCIENCE PROJECT REPORT ON FIFA20
## BUSINESS CASE: BASED ON THE FEATURES OF DATA WE NEED TO CLUSTER (GROUP) THE PLAYER BASE ON THEIR SKILLSET

### Abstract:
The FIFA 20 project aims to develop a robust player clustering model using machine learning techniques. The primary objective is to create a tool that can effectively group football players based on their skillsets and attributes.
Methodology involves data preprocessing, exploratory data analysis (EDA), feature engineering, and the application of machine learning algorithms. We used clustering techniques such as K-means clustering to group players based on their playing style, skills, and other relevant attributes.

### Device Project In-to Multiple Steps:
1.	Data Collection 
2.	Loading data
3.	Domain Analysis 
4.	.Basic Checks of data
5.	EDA (Univariate, Bivariate, Multivariate Analysis)
6.	Data Pre-processing 
7.	Feature Selection 
8.	Building ML Model
9.	Training & Model Evaluation
10.	Model Savings

### Loading data:
load data in python using pandas library

### Domain Analysis
Understanding the meaning of each feature and understanding the significance of each attribute in defining a player's skillset is essential.
**1.SOFIFA_ID:**
- It is a Unique Feature.Unique ID assigned to each football player on Sofifa.

**2. PLAYER_URL:**
- Player_url represents the URL or web link to the player's profile on Sofifa's website.

**3. SHORT_NAME:**
- It is player's short or commonly used name.

**4. LONG_NAME:**
-  Long_name represents player's formal identification. player's full or complete name. It includes the player's first name and last name.

**5. AGE:**
- Age Represents the player's age, typically measured in years.

**6. DOB (DATE OF BIRTH):**
- This Feature represents the date of birth of the football player. It indicates the player's birthdate, providing information about their age.

**7. HEIGHT_CM (HEIGHT IN CENTIMETERS):**
- This feature represents the player's height in centimeters.

**8. WEIGHT_KG (WEIGHT IN KILOGRAMS):**
- This Feature indicates the player's weight in kilograms. weight is a also physical attribute that can motivats a player's performance and role on the field.

**9. NATIONALITY (NATIONALITY OF THE PLAYER):**
- This Feature Represents the nationality or country of  the football player.

**10. CLUB (FOOTBALL CLUB):**
- This feature represent the name of the football club to which the player belongs.

**11. OVERALL:**
- The "overall" feature represents the player's overall skill rating.

**12. POTENTIAL:**
- The potential feature represent a player's potential skill rating. It represents the estimated maximum skill level a player can achieve in the future, considering factors like age, development, and performance.

**13. VALUE (IN EUROS):**
- The value_eur Feature represents the market value of the player in Euros.

**14. WAGE (IN EUROS):**
- The wage_eur Feature represents the player's weekly wage in Euros. It represents the salary that the player earns from their club on a weekly basis.

**15. PLAYER POSITIONS:**
- The player_positions Feature represents the positions on the football field where the player is capable of playing.

**16. PREFERRED FOOT:**
- This Feature reprsents whether a football player prefers to kick the ball with their left foot or right foot.

**17. INTERNATIONAL REPUTATION:**
- This attribute measures a player's reputation on the international stage.

**18. WEAK FOOT:**
- A weak foot just means that your gameplay is better when you use one foot instead of the other.As a football player can be either right-footed or left-footed.Ratings range from 1 to 5, with 1 indicating a very weak weaker foot and 5 indicating a very strong weaker foot.

**19. SKILL MOVES:**
-  Skill Moves feature represent the actions like controlling the ball with a first touch, developing into more complex actions like scoring an overhead kick from a cross.Ratings range from 1 to 5, with 1 representing limited skill moves and 5 indicating a high level of skillfulness.

**20. WORK RATE:**
- a player's work rate in football refers to how hard they work and how much they move around during a game. It's about how actively they participate in the match.

**21. BODY_TYPE:**
- This column represents the player's body type.

**22. REAL_FACE:**
- This Feature indicates whether the player has a real face in the game.

**23. RELEASE_CLAUSE_EUR:**
- This column indicates the release clause value in Euros for the player.A release clause is a set fee agreed upon when a player signs a contract with a club, allowing another club to sign that player if the fee is met.

**24. PLAYER_TAGS:**
- These Player tags can provide information about specific characteristics or qualities of the player. For example, a player might have tags like "Speedster," "Clinical Finisher," "Playmaker," or "Physical Presence."

**25. TEAM_POSITION:**
- This Feature represents the player's position within their team's formation or lineup.

**26. TEAM_JERSEY_NUMBER:**
- This column represents the jersey number worn by the player within their current team.

**27.LOANED_FROM:**
- If a player is currently on loan from another club, this column indicates the name of the club from which the player is loaned.

**28. JOINED:**
- This column represents the date when the player officially joined their current club.

**29. CONTACT VALID UNTIL:**
- This Feature indicates the end year of the player's contract with their current club.

**30. NATION POSITION:**
- Nation Position is a term that Represent the role or position a football player plays when representing their home country's national team.

**31. NATION JERSEY NUMBER:**
- This Feature represents the jersey number worn by the player when representing their national team.

**32. PACE:**
- The Pace Feature is related to a player's speed and acceleration on the football field.

**33. SHOOTING:**
- Shooting represent trying to kick or head the football into the goal to score a point. It's like trying to throw a ball using your feet or head in football.

**34. PASSING:**
- Passing is like sharing the ball with teammates. It's when a player kicks the ball to another player on the same team to help move the ball forward on the field.

**35. DRIBBLING:**
- Dribbling is like keeping the ball close to feet while you run with it.In football, it's about using your feet to keep control of the ball while you run and dodge opponents.

**36. DEFENDING:**
- Defending is all about stopping the other team from scoring.

**37. PHYSIC:**
- In football and games like FIFA 20, PHYSIC tells us how strong and hardworking a player is on the field.

**38. GK DIVING:**
- GK Diving is a skill that goalkeepers use in football. It's like a goalkeeper's superpower for jumping and diving quickly to stop the ball from going into the goal.

**39. GK HANDLING:**
- GK Handling is represent how good a goalkeeper is at safely catching the ball when it's kicked toward them.

**40. GK KICKING:**
- GK Kicking  is a skill that goalkeepers have to kick the ball to different places on the field.

**41. GK REFLEX:**
- GK Reflex is represent all about a goalkeeper's ability to react lightning-fast to unexpected situations and make amazing saves.

**42. GK SPEED:**
- GK Speed is represent all about how fast the goalkeeper can move around the goal area.

**43. GK POSITIONING:**
- GK Positioning is represent about them being in the right place at the right time to block shots.

**44. PLAYER TRAITS:**
- Player Traits are like a player's special habits. These are actions they're more likely to do in matches.

**45. ATTACKING CROSSING:**
- It's like a player trying to kick the ball from the sides of the field towards the middle, where their teammates are waiting to score a goal.

**46. ATTACKING FINISHING:**
- Attacking Finishing is about creating a plan to score a goal.

**47. ATTACKING HEADING ACCURACY:**
- Attacking Heading Accuracy is indicate a player using their head to hit the football and score a goal.

**48. ATTACKING SHORT PASSING:**
- Attacking Short Passing is like a player's ability to pass the football quickly and accurately over a short distance to their teammates.

**49. ATTACKING VOLLYES:**
- Attacking Vollyes as kicking the football while it's up in the air, without letting it touch the ground first.

**50. SKILL DRIBBLING:**
- Skill Dribbling is all about a player's ability to show off fancy and tricky moves while they're running with the ball.

**51. SKILL CURVE:**
- When a player tries to kick the ball in a way that makes it curve or bend, that's called Skill Curve.

**52. SKILL FK ACCURACY:**
- Skill FK Accuracy is player kicking the ball from a set distance, like aiming to score a goal directly from a free kick.

**53. SKILL LONG PASSING:**
- Kill Long Passing is about how well a player can pass the ball over long distances.

**54. SKILL BALL CONTROL:**
- Skill Ball Control is how skilled a player is at handling the ball with their feet during the game.

**55. MOVEMENT ACCELARATION:**
- Movement Accelaration measures how fast a player can speed up when they're trying to reach their maximum running speed.

**56. MOVEMENT SPRINT SPEED:**
- Movement Sprint Speed is all about how fast a player can run at their top speed.

**57. MOVEMENT AGILITY:**
- Movement Agility is about how  player ability to dash in different directions without tripping or losing their balance.

**58. MOVEMENT REACTION:**
- Movement Reaction is represent how fast a player can react to things happening on the field. It's like measuring how quickly they can respond to what's going on during the game.

**59. MOVEMENT BALANCE:**
- Movement Balance represent how well they can keep themselves steady and not fall over when they're running and moving.

**60. SHOT POWER:**
- Shot Power is about measuring how hard they can kick the ball to make it go fast and strong toward the goal.

**61. POWER JUMPING:**
- Power Jumping  assessing their ability to leap into the air, which can be useful for heading the ball or reaching high places.

**62. POWER STAMINA:**
- Power Stamina measuring how much energy they have to keep running and playing effectively.

**63. POWER STRENGTH:**
- Power Strength is about how strong they are and how well they can use their strength in different situations during the game.

**64. POWER LONG SHOT:**
- Power Long Shot is the how good they are at making strong and accurate shots from far away.

**65. MENTALITY AGGRESSION:**
- Mentality Aggression relates to how determined and assertive they are in their play, which can impact their performance.

**66. MENTALITY INTERCEPTION:**
- Mentality Interception is about trying to steal the ball or take possession away from the other team.

**67. MENTALITY POSITIONING:**
- Mentality Positioningis like deciding how your players should play in terms of defense or attack.

**68. MENTALITY VISION:**
- Mentality Vision refers to a player's mental perspective and awareness of what's happening during the match.

**69. MENTALITY PENALTIES:**
- Mentality Penalties In football, when a player breaks the rules (commits an infraction), the referee may call a foul and assess a penalty against the player or their team.

**70. MENTALITY COMPOSURE:**
- Mentality Composure is about a player's ability to handle pressure and stay focused, especially after making a mistake.

**71. MENTALITY COMPOSURE:**
- Mentality Composure is like a mental process in sports. It involves recognizing that you've made a mistake, then regrouping your thoughts, and refocusing on the game.

**72. DEFENDING MAKING:**
- Defending Making is like a strategy to make the field smaller and limit the options of the opposing team.

**73. DEFENDING STANDING TACKLE:**
- Defending Standing Tackle is when a player uses their feet to make a tackle and take the ball away from the opponent while staying on their feet.

**74. DEFENDING SLIDING TACKLE:**
- Defending Sliding Tackle is when a player slides on the ground to tackle and take the ball away from the opponent.

**75. GOALKEEPER HANDLING:**
- Goalkeeper Handling represent a goalkeeper, handle the ball with hands but only within penalty area. Goalkeeper Handling refers to how skillfully a goalkeeper can catch or grip the ball safely when it comes toward them.

**76. GOALKEEPER KICKING:**
- Goalkeepers often kick the ball to restart play or distribute it to teammates. "GOALKEEPING KICKING" assesses the accuracy and power of a goalkeeper's kicks.

**77. GOALKEEPER POSITIONAING:**
- Goalkeeper Positionaing refers to where a goalkeeper positions themselves in the goal area to make it harder for the opposing team to score.

**78. GOALKEEPER REFLEX:**
- Goalkeeper Reflex is about a goalkeeper's ability to react quickly to unexpected situations, like fast shots or deflections.
#### THE REMAINING FEATURE IS THE ABBREVATION OF FOOTBALL POSITION SCORE:
**LS:**
Long snapper or left striker.

**ST:**
Striker

**RS:**
Right striker

**LW:**
Left sided wingers.

**LF:**
Left forword

**CF:**
Center forword

**RF:**
Right forword

**RW:**
The RW is usually on the right end of the attacking trident, with the Striker and Left Winger, which mainly contributes to the team in terms of goals and assists.

**LAM:**
Left attacking midfield

**CAM:**
Center attacking midfield

**RAM:**
Right attacking midfield

**LM:**
Left midfield

**LCM:**
Left center midfield

**CM:**
Center Midfield

**RCM:**
Right center midfield

**RM:**
Right midfield

**LWB:**
Left Wing Back

**LDM:**
Left defensive midfield

**CDM:**
Center defensive midfield

**RDM:**
Right defensive midfield

**RWB:**
Right wing back

**LB:**
Left back

**LCB:**
Left center back

**CB:**
Center back

**RCB:**
Right center back

**RB:**
Right back

### EDA (Univariate, Bivariate, Multivariate Analysis)
#### 1.	Univariate Data Analysis
Use sweetviz library and generate a html report of all feature to do univariate analysis, in that we get the Minimum, Maximum, Some statistical information of the particular feature.

#### 2.	Bivariate Data Analysis
In Bivariate analysis we check the relation of independent features to each other.

#### 3.	Multivariate Data Analysis¶
In Multivariate analysis we check the relation of two independent features to each other.

### Here Some Condition & Plotting
#### 1.prepare a rank ordered list of top 10 countries with most players. Which countries are producing the most footballers that play at this level?
![image](https://github.com/KhairnarRutuja/FIFA-20/assets/135214279/4c6f0aa2-66cb-4a16-8bcb-2b7509112184)

**Observation/Insights**
- Belgium country has the highest representation player among the top 10.

#### 2.plot the distribution of overall rating vs. Age of players. Interpret what is the age after which a player stops improving?
![image](https://github.com/KhairnarRutuja/FIFA-20/assets/135214279/36ea230b-026a-4216-88c2-c05fb04c0ee0)

**Observation/Insights**
- It can be estimated that a player typically stops improving after the age of 40.

#### 3.which type of offensive player tends to get paid the most: the striker, the right winger, the left winger?
![image](https://github.com/KhairnarRutuja/FIFA-20/assets/135214279/770e1186-868e-4a78-8c0e-b28bacfd19d3)

**Observation/Insights**
- The graph shows that left wingers have a higher salary, increasing 20,000, compared to both right wingers and strikers. Right wingers also tend to have relatively high salaries when compared to strikers.

### Data Preprocessing
* First we check the missing values, and then check missing values in percentage, we seen that the above 50% to 90% missing value and some unique feature also contain missing value so we drop this feature.
* Remaining feature missing value is less than 50% so we impute the missing value with median and mode.
* Second we Handle categorical data and use Manual encoding and frequency encoding. Because features has contain lots of label.
* In this data I’m Clearly seen that some feature has lots of outlier & we impute them, for that first we check the distribution of all feature and plot the box plot and decide the technique. In this data we are handle only important feature outlier, because the remaining feature is unique or some feature is not required to handle outlier.
* Scale the numerical independent feature with the help of Minmax scalar and scale the feature. Use min max scaling because of dataset contain large amount of outlier so outlier is going to be biased.

### Feature Scaling
* First we drop unique and constant feature Here we are going to drop unique column as well as lots of missing value column. The column ls,st,rs,lw,cf etc.. are playing position in the game and the data in this columns is basically the potential of the player if were to play in that position, so we assume the player only plays with the team position and we will drop this column.
* Check the correlation with the help of heatmap and seen the From the above heatmap is very difficult to find highly correlated feature so we are create a python code to check the highly corelated feature and drop highly correlated feature. 
* The dataset not contain any duplicates.
* After that save & load the preprocess data, Save the dataframe to a CSV file.
* Using PCA(PRINCIPLE COMPONENET ANALYSIS)  to reduce the feature. Here we are select 10 components because less variance loss

### Model Creation & Evaluation
* Define Independent variable.
* Use K-Means Clustering algorithm using elbow Method technique to determining the optimal number of clusters (k) in a K-means clustering algorithm. From the plot we are select 3 cluster because of odd no and more variance.
U* sing K-Means Clustering to get a Silhouette Score is  0.62.

### Model Saving
Save the model using pickle file.

### Conclusion
In this FIFA20 data science project, we embarked on a journey to create a player clustering model based on their skillsets and attributes. Through a systematic approach encompassing data collection, exploratory data analysis (EDA), data preprocessing, and clustering, we gained valuable insights into the virtual footballing world. We successfully applied the K-Means clustering algorithm and determined that three clusters provide a meaningful grouping of players. This clustering method allows us to categorize players into distinct segments based on their skillsets. Through multivariate data analysis, we observed several intriguing relationships between player attributes. For instance, players who exhibit a preference for their right foot tend to have distinct patterns in various attributes, including dribbling, passing, and potential. Belgium emerged as the country with the highest representation of players in the top 10 countries. This finding sheds light on the diversity of nationalities in the FIFA20 player database. Our analysis suggests that players typically stop improving after the age of 40. This insight can be valuable for player development strategies in the game. Left wingers tend to have higher salaries compared to right wingers and strikers, indicating that this offensive player type is compensated more generously in the virtual football world. We meticulously handled missing values, outliers, and categorical data, ensuring the robustness of our clustering model. Feature scaling and dimensionality reduction through PCA enhanced the model's performance. The clustering model can provide valuable market insights by identifying trends in player attributes and preferences. This information can guide decisions related to player pricing and availability in the virtual transfer market.



