# 2026 FIFA World Cup Match Predictions

This is the 2026 FIFA World Cup Prediction project! This project utilizes historical football match data (since 1872) and machine learning to forecast the outcomes of all scheduled group stage matches for the 2026 tournament.

## Model Overview
Engineered a custom feature set to represent dynamic team strength and match contextual factors:
1. **Dynamic Elo Ratings**: Calculated chronologically across the entire historical match dataset to ensure stable and mature team strength indicators.
2. **Win Probabilities**: Extracted the expected win probability based on the Elo difference between competing teams (including a home advantage adjustment of 200 Elo points).
3. **Friendly Flag**: Identifies friendly matches, which represent higher variance due to tactics testing and squad rotation.
4. **Neutral Ground Flag**: Denotes if a game is hosted on neutral ground.

### Model Metrics
We optimized the model by tuning a **Gradient Boosting Classifier** to capture complex non-linear feature interactions:
- **Baseline Accuracy**: 46.13%
- **Optimized Model Test Accuracy**: **53.31%** (a **+7.18% absolute boost** over the baseline)

All model training and feature engineering details can be explored in the [Model Training Notebook](notebooks/models_training.ipynb).

---

## Group Stage Predictions

Below are the predicted outcomes and class probabilities for all 70 group stage matches, generated using our tuned Gradient Boosting model.

| Date | Matchup | Predicted Outcome | Prob. Team A Win | Prob. Draw | Prob. Team B Win | Venue |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Jun 12, 2026 | Canada vs. Bosnia and Herzegovina | **Canada Win** | 62.5% | 23.0% | 14.5% | Toronto, Canada |
| Jun 12, 2026 | United States vs. Paraguay | **United States Win** | 36.8% | 30.8% | 32.4% | Inglewood, United States |
| Jun 13, 2026 | Qatar vs. Switzerland | **Switzerland Win** | 11.0% | 20.6% | 68.3% | Santa Clara, United States |
| Jun 13, 2026 | Brazil vs. Morocco | **Brazil Win** | 69.3% | 18.3% | 12.4% | East Rutherford, United States |
| Jun 13, 2026 | Haiti vs. Scotland | **Scotland Win** | 25.4% | 30.4% | 44.2% | Foxborough, United States |
| Jun 13, 2026 | Australia vs. Turkey | **Australia Win** | 40.2% | 24.0% | 35.9% | Vancouver, Canada |
| Jun 14, 2026 | Germany vs. Curaçao | **Germany Win** | 69.9% | 21.2% | 8.8% | Houston, United States |
| Jun 14, 2026 | Ivory Coast vs. Ecuador | **Ecuador Win** | 31.3% | 28.4% | 40.3% | Philadelphia, United States |
| Jun 14, 2026 | Netherlands vs. Japan | **Netherlands Win** | 46.6% | 24.2% | 29.3% | Arlington, United States |
| Jun 14, 2026 | Sweden vs. Tunisia | **Sweden Win** | 39.7% | 24.8% | 35.5% | Guadalupe, Mexico |
| Jun 15, 2026 | Belgium vs. Egypt | **Belgium Win** | 53.5% | 28.4% | 18.0% | Seattle, United States |
| Jun 15, 2026 | Iran vs. New Zealand | **Iran Win** | 69.9% | 18.0% | 12.1% | Inglewood, United States |
| Jun 15, 2026 | Spain vs. Cape Verde | **Spain Win** | 69.2% | 24.2% | 6.5% | Atlanta, United States |
| Jun 15, 2026 | Saudi Arabia vs. Uruguay | **Uruguay Win** | 13.0% | 27.8% | 59.2% | Miami Gardens, United States |
| Jun 16, 2026 | France vs. Senegal | **France Win** | 45.5% | 24.2% | 30.3% | East Rutherford, United States |
| Jun 16, 2026 | Iraq vs. Norway | **Norway Win** | 29.3% | 29.7% | 41.1% | Foxborough, United States |
| Jun 16, 2026 | Argentina vs. Algeria | **Argentina Win** | 66.5% | 24.9% | 8.6% | Kansas City, United States |
| Jun 16, 2026 | Austria vs. Jordan | **Austria Win** | 47.4% | 24.6% | 28.0% | Santa Clara, United States |
| Jun 17, 2026 | Portugal vs. DR Congo | **Portugal Win** | 71.5% | 19.5% | 9.0% | Houston, United States |
| Jun 17, 2026 | Uzbekistan vs. Colombia | **Colombia Win** | 19.4% | 23.4% | 57.2% | Mexico City, Mexico |
| Jun 17, 2026 | England vs. Croatia | **England Win** | 38.6% | 25.3% | 36.1% | Arlington, United States |
| Jun 17, 2026 | Ghana vs. Panama | **Panama Win** | 24.9% | 20.6% | 54.5% | Toronto, Canada |
| Jun 18, 2026 | Switzerland vs. Bosnia and Herzegovina | **Switzerland Win** | 69.3% | 18.3% | 12.4% | Inglewood, United States |
| Jun 18, 2026 | Canada vs. Qatar | **Canada Win** | 64.1% | 25.4% | 10.5% | Vancouver, Canada |
| Jun 18, 2026 | Czech Republic vs. South Africa | **Czech Republic Win** | 60.1% | 17.6% | 22.2% | Atlanta, United States |
| Jun 18, 2026 | Mexico vs. South Korea | **Mexico Win** | 54.8% | 24.9% | 20.4% | Zapopan, Mexico |
| Jun 19, 2026 | Turkey vs. Paraguay | **Turkey Win** | 38.6% | 25.3% | 36.1% | Santa Clara, United States |
| Jun 19, 2026 | United States vs. Australia | **Australia Win** | 28.2% | 30.0% | 41.8% | Seattle, United States |
| Jun 19, 2026 | Scotland vs. Morocco | **Morocco Win** | 26.1% | 25.5% | 48.4% | Foxborough, United States |
| Jun 19, 2026 | Brazil vs. Haiti | **Brazil Win** | 81.4% | 14.4% | 4.2% | Philadelphia, United States |
| Jun 20, 2026 | Germany vs. Ivory Coast | **Germany Win** | 56.1% | 19.4% | 24.5% | Toronto, Canada |
| Jun 20, 2026 | Ecuador vs. Curaçao | **Ecuador Win** | 76.4% | 18.2% | 5.4% | Kansas City, United States |
| Jun 20, 2026 | Netherlands vs. Sweden | **Netherlands Win** | 61.6% | 24.2% | 14.2% | Houston, United States |
| Jun 20, 2026 | Tunisia vs. Japan | Draw | 21.9% | 41.2% | 36.9% | Guadalupe, Mexico |
| Jun 21, 2026 | Uruguay vs. Cape Verde | **Uruguay Win** | 50.3% | 38.3% | 11.4% | Miami Gardens, United States |
| Jun 21, 2026 | Spain vs. Saudi Arabia | **Spain Win** | 61.8% | 21.6% | 16.6% | Atlanta, United States |
| Jun 21, 2026 | Belgium vs. Iran | **Belgium Win** | 38.8% | 25.2% | 35.9% | Inglewood, United States |
| Jun 21, 2026 | New Zealand vs. Egypt | Draw | 21.9% | 41.2% | 36.9% | Vancouver, Canada |
| Jun 22, 2026 | Jordan vs. Algeria | **Algeria Win** | 26.0% | 29.7% | 44.3% | Santa Clara, United States |
| Jun 22, 2026 | Argentina vs. Austria | **Argentina Win** | 68.4% | 24.4% | 7.2% | Arlington, United States |
| Jun 22, 2026 | Norway vs. Senegal | Draw | 21.9% | 41.2% | 36.9% | East Rutherford, United States |
| Jun 22, 2026 | France vs. Iraq | **France Win** | 50.3% | 38.3% | 11.4% | Philadelphia, United States |
| Jun 23, 2026 | Panama vs. Croatia | Draw | 20.5% | 44.8% | 34.7% | Toronto, Canada |
| Jun 23, 2026 | Portugal vs. Uzbekistan | **Portugal Win** | 71.0% | 16.6% | 12.4% | Houston, United States |
| Jun 23, 2026 | Colombia vs. DR Congo | **Colombia Win** | 45.8% | 43.8% | 10.4% | Zapopan, Mexico |
| Jun 23, 2026 | England vs. Ghana | **England Win** | 72.6% | 18.8% | 8.7% | Foxborough, United States |
| Jun 24, 2026 | Bosnia and Herzegovina vs. Qatar | **Bosnia and Herzegovina Win** | 54.0% | 23.2% | 22.8% | Seattle, United States |
| Jun 24, 2026 | Canada vs. Switzerland | Draw | 25.1% | 48.7% | 26.2% | Vancouver, Canada |
| Jun 24, 2026 | Morocco vs. Haiti | **Morocco Win** | 61.6% | 24.2% | 14.2% | Atlanta, United States |
| Jun 24, 2026 | Mexico vs. Czech Republic | **Mexico Win** | 61.4% | 23.7% | 14.9% | Mexico City, Mexico |
| Jun 24, 2026 | Scotland vs. Brazil | **Brazil Win** | 9.6% | 12.9% | 77.5% | Miami Gardens, United States |
| Jun 24, 2026 | South Africa vs. South Korea | **South Korea Win** | 21.4% | 24.5% | 54.1% | Guadalupe, Mexico |
| Jun 25, 2026 | Curaçao vs. Ivory Coast | **Ivory Coast Win** | 6.8% | 14.3% | 78.9% | Philadelphia, United States |
| Jun 25, 2026 | Paraguay vs. Australia | **Australia Win** | 29.3% | 29.7% | 41.1% | Santa Clara, United States |
| Jun 25, 2026 | United States vs. Turkey | **Turkey Win** | 28.8% | 28.6% | 42.6% | Inglewood, United States |
| Jun 25, 2026 | Japan vs. Sweden | **Japan Win** | 56.1% | 19.4% | 24.5% | Arlington, United States |
| Jun 25, 2026 | Tunisia vs. Netherlands | **Netherlands Win** | 23.8% | 27.2% | 49.0% | Kansas City, United States |
| Jun 25, 2026 | Ecuador vs. Germany | **Ecuador Win** | 49.5% | 21.9% | 28.6% | East Rutherford, United States |
| Jun 26, 2026 | Uruguay vs. Spain | **Spain Win** | 26.1% | 25.5% | 48.4% | Zapopan, Mexico |
| Jun 26, 2026 | Cape Verde vs. Saudi Arabia | **Cape Verde Win** | 39.7% | 24.8% | 35.5% | Houston, United States |
| Jun 26, 2026 | Norway vs. France | **France Win** | 19.4% | 23.4% | 57.2% | Foxborough, United States |
| Jun 26, 2026 | Egypt vs. Iran | **Iran Win** | 21.4% | 37.6% | 41.0% | Seattle, United States |
| Jun 26, 2026 | New Zealand vs. Belgium | **Belgium Win** | 19.4% | 23.4% | 57.2% | Vancouver, Canada |
| Jun 26, 2026 | Senegal vs. Iraq | **Senegal Win** | 61.6% | 24.2% | 14.2% | Toronto, Canada |
| Jun 27, 2026 | Algeria vs. Austria | **Algeria Win** | 37.1% | 31.0% | 31.9% | Kansas City, United States |
| Jun 27, 2026 | Panama vs. England | **England Win** | 23.8% | 27.2% | 49.0% | East Rutherford, United States |
| Jun 27, 2026 | Jordan vs. Argentina | **Argentina Win** | 3.6% | 4.1% | 92.3% | Arlington, United States |
| Jun 27, 2026 | DR Congo vs. Uzbekistan | **Uzbekistan Win** | 27.9% | 30.3% | 41.9% | Atlanta, United States |
| Jun 27, 2026 | Colombia vs. Portugal | **Colombia Win** | 39.9% | 24.4% | 35.6% | Miami Gardens, United States |
| Jun 27, 2026 | Croatia vs. Ghana | **Croatia Win** | 69.9% | 18.0% | 12.1% | Philadelphia, United States |

---

## How to Run the Predictions
1. Initialize the virtual environment and install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Open and run the predictions notebook:
   - [2026 World Cup Predictions Notebook](notebooks/world_cup_2026_predictions.ipynb)
