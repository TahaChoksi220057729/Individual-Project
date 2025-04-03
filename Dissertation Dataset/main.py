import nltk
import spacy
import os
import pprint
from speech_evaluator import MUNSpeechEvaluator

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load SpaCy model
try:
    nlp = spacy.load("en_core_web_md")
except:
    # If model isn't available, download it
    os.system("python -m spacy download en_core_web_md")
    nlp = spacy.load("en_core_web_md")

def example_usage():
    """Example demonstrating how to use the MUN Speech Evaluator."""
    # Sample training data (in a real application, this would be loaded from a database or file)
    sample_speeches = ["""Honorable Chair, distinguished delegates, and esteemed colleagues, I stand before you today as the Delegate of China to address the urgent global challenge of climate change.China firmly believes that climate change is the paramount threat of our time, requiring immediate collaborative actions by all nations. Our nation is deeply committed to adopting sustainable development practices and investing in renewable energy sources to counter the risks posed by climate change. We have observed that rapid industrial growth, if unchecked, can exacerbate climate change while undermining our collective efforts toward sustainability.China has implemented numerous policies that promote energy efficiency, environmental protection, and technological innovations to mitigate climate change impacts. We recognize that climate change is a multifaceted crisis involving economic, social, and environmental dimensions that require integrated solutions. Through strategic investments in clean technologies, China is reducing its carbon footprint and ensuring that economic progress does not worsen climate change. We are dedicated to sharing our experiences and best practices with the global community, thus strengthening international collaboration on climate change mitigation. Climate change is not confined to national borders, and our actions must transcend geographic and political boundaries to achieve sustainable development. Our nation advocates for comprehensive global frameworks that incentivize sustainable methods of development while addressing the severe impacts of climate change. We believe that transparency, cooperation, and innovation are essential to ensuring that climate change does not compromise future generations' well-being. China has actively participated in multilateral initiatives and partnerships that aim to alleviate the destructive consequences of climate change. Investments in renewable energy, reforestation programs, and sustainable urban planning are crucial to combating the pervasive threat of climate change. We call on every nation to strengthen its commitments under international agreements and prioritize sustainable policies to curb climate change. Climate change demands that we rethink traditional development models and adopt holistic approaches that promote environmental resilience. Our delegation recognizes that only through collective resolve and shared technological progress can we effectively mitigate climate change. We urge developed and developing countries alike to join forces in crafting policies that balance economic growth with the imperative to halt climate change. In addition to policy reforms, public awareness and education are vital to fostering a global culture that resists the adverse effects of climate change. The challenges posed by climate change are vast, yet our combined determination will guide us toward innovative and sustainable solutions. China is prepared to lead by example, forging partnerships that drive both economic development and the urgent need to fight climate change. Together, we must act decisively to transform our energy systems, bolster climate resilience, and secure a sustainable future for all citizens. I thank you for your attention and call upon every nation to work relentlessly towards a world where climate change is effectively managed through sustainable development.
                       """,
                       """Honorable Chair, distinguished delegates, and esteemed colleagues, I stand before you as the Delegate of the United States to address the critical challenges posed by climate change. Climate change is a global crisis that threatens economic stability, public health, and the environment, demanding decisive international cooperation. Our nation has taken significant strides to combat climate change by investing in renewable energy, technological innovation, and sustainable infrastructure. The United States firmly believes that climate change can be mitigated through collective commitment and innovative policy solutions that promote sustainable development. We recognize that climate change affects every nation, transcending borders and socioeconomic boundaries while imposing severe risks on vulnerable communities. American initiatives, such as clean energy research and emissions reduction targets, are designed to reduce the adverse impacts of climate change. Our delegation advocates for increased transparency, accountability, and financial support to help developing nations adapt to climate change. Through multilateral agreements and strategic partnerships, the United States is committed to strengthening global resilience against climate change. We emphasize that climate change is not merely an environmental issue, but a catalyst that influences economic growth, security, and social progress. Our policies focus on creating green jobs, enhancing energy efficiency, and fostering innovation in renewable technologies to mitigate climate change. The United States calls on every nation to prioritize climate change in their national agendas, investing in sustainable methods of development. Addressing climate change will also require rethinking traditional energy models and transitioning to a low-carbon future. In conclusion, the United States is prepared to lead by example and work collaboratively with all nations to combat climate change. We believe that our shared determination and innovative spirit can transform global challenges into opportunities for sustainable progress. Together, we can create a future where climate change is managed effectively and every nation prospers in a cleaner, safer world. I thank you for your attention and encourage decisive action to overcome the challenges posed by climate change.
                       """,
                       """Honorable Chair, distinguished delegates, and esteemed colleagues, I am honored to represent Mali in addressing the profound impacts of climate change on our communities and the world. Climate change remains an urgent challenge that undermines food security, water availability, and the overall well-being of our people. In Mali, the effects of climate change are evident in desertification, irregular rainfall, and increased vulnerability of our rural populations. We believe that proactive measures and sustainable development strategies are essential to confront climate change and its many repercussions. Mali is committed to implementing adaptive policies that address climate change while promoting environmental resilience. Our initiatives focus on sustainable land management, reforestation, and renewable energy projects to reduce the impact of climate change on our natural resources. We recognize that climate change exacerbates poverty and inequality, making international support and cooperation indispensable for our progress. The challenges of climate change demand a collective response, and Mali is ready to contribute by sharing traditional knowledge and innovative practices. Our delegation advocates for a renewed global commitment to sustainable development, where climate change is addressed through inclusive policies and strategic investments. Climate change has become a driving force behind our efforts to secure a sustainable future for our citizens. We call upon the international community to offer financial assistance, capacity building, and technological support to nations affected by climate change. By working together, we can implement robust strategies that mitigate climate change while fostering long-term economic growth. In conclusion, Mali stands in solidarity with all nations in the battle against climate change. We believe that by embracing sustainable methods of development and reinforcing global partnerships, we can overcome the threats posed by climate change. I thank you for your attention and urge every delegate to work with determination and compassion to ensure a resilient future for our planet and its people.
                       """,
                       """Honorable Chair, esteemed delegates, and distinguished colleagues, I stand before you as the Delegate of the United States to discuss climate change, an existential crisis that requires global unity. Climate change is not a distant threat; it is a present reality, affecting economies, ecosystems, and human lives. The United States recognizes that climate change exacerbates existing vulnerabilities, leading to food insecurity, water scarcity, and extreme weather events. As a nation, we have taken bold steps to mitigate climate change by advancing clean energy, enhancing carbon reduction policies, and driving technological innovation. Our commitment to climate change action is evident through investments in wind and solar energy, electric vehicle infrastructure, and emission reduction targets. However, climate change cannot be addressed by one nation alone—it requires global cooperation and shared responsibility. The United States calls upon all nations to implement ambitious climate policies, support sustainable economic development, and foster climate resilience. Climate change has already caused rising sea levels, devastating wildfires, and prolonged droughts, underscoring the urgent need for action. We must collectively accelerate the transition to a low-carbon economy while ensuring that developing nations receive the necessary financial and technological support to combat climate change. In conclusion, the United States urges all delegates to prioritize climate change in policymaking and international cooperation. Our planet’s future depends on immediate and decisive action against climate change. Let us work together to safeguard the environment and secure a more resilient world for future generations.
                       """,
                       """Honorable Chair, distinguished delegates, and esteemed colleagues, the United States stands before you to address one of the most pressing challenges of our time: climate change. Climate change is an undeniable global crisis, disrupting weather patterns, threatening food supplies, and increasing the frequency of natural disasters. Every nation, regardless of economic standing, must recognize that climate change is a fundamental threat to long-term security and stability. The United States has taken strong steps to mitigate climate change by investing in renewable energy, reducing fossil fuel dependency, and committing to carbon neutrality. However, climate change remains an issue that requires immediate, collective action from all nations. The economic, social, and environmental costs of inaction far outweigh the investments required to transition towards sustainability. We urge nations to accelerate green energy development, enforce climate policies, and commit to meaningful emissions reduction. The effects of climate change—rising sea levels, unpredictable weather, and environmental degradation—demand innovative solutions and sustained global cooperation. The United States remains committed to leading the fight against climate change while ensuring economic prosperity and environmental justice. In closing, climate change is a defining challenge of the 21st century, and we must act before it is too late. Let us join forces to combat climate change, safeguard our planet, and create a sustainable future for all.
                       """,
                       """Honorable Chair, esteemed delegates, and distinguished colleagues, the United States recognizes climate change as a major global challenge. Climate change affects economies, security, and public health, requiring urgent international action. As temperatures rise and natural disasters become more frequent, the world must unite to combat climate change through sustainable policies. The United States has committed to reducing carbon emissions, increasing renewable energy usage, and investing in climate research. However, climate change remains a shared responsibility, and all nations must take bold steps to limit its impact. We encourage nations to adopt renewable energy, strengthen climate resilience, and invest in adaptation strategies. Climate change is a crisis that threatens food supplies, coastal cities, and biodiversity, and we must act decisively. In conclusion, climate change demands cooperation and leadership. The United States urges global action to prevent long-term damage and protect future generations.
                       """,
                       """Honorable Chair, distinguished delegates, and esteemed colleagues, the United States acknowledges climate change as an urgent issue requiring international cooperation. Climate change has already intensified extreme weather, displaced communities, and damaged ecosystems. Without action, these consequences will worsen. The United States has prioritized addressing climate change through clean energy investments and carbon reduction policies. However, combating climate change requires more than national policies; it demands a global response. The transition to sustainable development is essential to reducing climate change risks. We urge nations to accelerate their climate commitments, invest in green technology, and reduce dependency on fossil fuels. Climate change threatens global security, and we must act before the damage is irreversible. In closing, the United States calls upon all nations to commit to strong climate policies. Together, we can fight climate change and secure a sustainable future.
                       """,
                       """Honorable Chair, distinguished delegates, and esteemed colleagues, the United States acknowledges the serious challenge posed by climate change. Climate change is an issue that affects all countries, disrupting economies and damaging the environment. As temperatures rise, we must take action to prevent further harm.The United States has committed to addressing climate change by investing in clean energy and reducing greenhouse gas emissions. However, climate change cannot be solved by one nation alone. It requires global cooperation and strong commitments from all countries.We encourage nations to work together to reduce carbon emissions and promote renewable energy. Climate change is a serious problem, and without action, its effects will only worsen.In conclusion, the United States urges the international community to act now. Climate change is a threat we cannot ignore.
                       """,
                       """Honorable Chair, esteemed delegates, climate change is one of the most important issues facing the world today. Climate change affects economies, health, and the environment, making it essential to act. The United States has taken steps to combat climate change by supporting renewable energy and improving energy efficiency. However, climate change requires more action. Nations must work together to prevent further harm. We call on all countries to take the issue of climate change seriously. Working together, we can develop solutions and reduce the damage being caused by climate change.In closing, climate change is a serious problem that we must address. The United States is committed to working with the world to fight climate change.
                       """,
                       """Honorable Chair, delegates, and colleagues, climate change is a big issue. It is affecting countries around the world, causing problems for people and the environment. Climate change must be addressed now before it gets worse. The United States is working to stop climate change by using more renewable energy and reducing pollution. But climate change is not a problem one country can solve alone. Other nations must also take action. We believe that working together is the best way to fight climate change. More countries need to focus on clean energy and reducing emissions. In conclusion, climate change is dangerous. The United States is ready to help fight climate change with the rest of the world.
                       """,
                       """Honorable Chair, climate change is a major problem. It affects many parts of the world and is getting worse. Climate change is causing problems for the environment and the economy. The United States is working to reduce climate change by using clean energy. We believe that stopping climate change is important for everyone. Countries must work together to solve the problem of climate change. The United States encourages other countries to take action. In conclusion, climate change is happening, and we need to stop it. The United States will continue to fight climate change.
                       """,
                       """Climate change is a problem. The world is getting hotter. Climate change is affecting everything. The United States is working on solutions. Climate change is dangerous. We need clean energy to stop climate change. More countries need to take climate change seriously. Climate change is happening fast. We must act. In conclusion, climate change is bad. We must stop climate change.
                       """,
                       """Honorable Chair, climate change is a global issue. Climate change is causing many problems. The United States is working on climate change. Climate change must be stopped. We need action. Countries need to help fight climate change. Climate change is affecting the world. We must do something. In conclusion, climate change is a big problem. The United States is trying to help.
                       """
                       ]
    
    # Sample scores (1-10 scale)
    sample_scores = [10, 10, 9, 9, 8, 8, 7, 7, 6, 5, 4, 3, 2]
    
    # Create and train evaluator
    evaluator = MUNSpeechEvaluator()
    evaluator.train(sample_speeches, sample_scores)
    
    # Evaluate a new speech
    new_speech = """Honorable Chair, esteemed delegates, and distinguished members of the Security Council, I stand before you as the Delegate of Slovenia to address the intertwined challenges of maritime security and climate change. Our oceans, vital for global trade and security, are increasingly threatened by maritime instability and the adverse effects of climate change. These dual challenges demand comprehensive international cooperation and innovative policy solutions. Maritime security remains a cornerstone of our collective safety and economic prosperity, as safe seas are essential for international commerce and peaceful coexistence. However, the growing impact of climate change is altering ocean currents, exacerbating extreme weather, and challenging our traditional maritime practices. Climate change is not only a threat to our coastal communities but also a catalyst for increased piracy, territorial disputes, and environmental degradation in our shared waters. In recent years, climate change has emerged as a critical factor that intensifies maritime security risks. Rising sea levels and intensified storm patterns, fueled by climate change, endanger maritime infrastructure and disrupt shipping routes. Slovenia is committed to addressing these threats by advocating for international frameworks that integrate maritime security with climate change adaptation measures. We believe that climate change and maritime security must be addressed in tandem, as neglecting one will undermine the effectiveness of our overall strategies. The challenges presented by climate change require robust cooperation among nations, with each country contributing to enhanced maritime surveillance, sustainable resource management, and disaster response. In this regard, Slovenia supports initiatives that leverage advanced technology and data sharing to monitor and protect our oceans from the adverse impacts of climate change. We call upon all member states to recognize that climate change and maritime security are mutually reinforcing issues that affect global peace and stability. Furthermore, Slovenia emphasizes the importance of environmental stewardship in safeguarding our maritime domains. Climate change has accelerated ocean acidification and habitat loss, thereby compromising marine biodiversity and destabilizing maritime ecosystems. Through collective action, we can mitigate these effects and promote resilient maritime security systems that adapt to the dynamic challenges posed by climate change. In conclusion, Slovenia urges the Security Council to adopt comprehensive policies that address both maritime security and the escalating impacts of climate change. Let us unite our efforts to ensure safe, stable, and sustainable seas for future generations. I thank you for your attention and call upon each nation to join us in this critical endeavor to secure our oceans and our planet against the intertwined threats of maritime insecurity and climate change.
    """
        
    score = evaluator.evaluate(new_speech)
    feedback = evaluator.generate_feedback(new_speech)
    
    print(f"Speech Score: {score}/10")
    print("\nDetailed Feedback:")
    
    # Content section
    print("Content:")
    print(f"  Word count: {feedback['content']['word_count']}")
    print(f"  Recommended word count: {feedback['content']['recommended_word_count']}")
    
    # Print entities by category
    print("  Entity Analysis:")
    for category, entities in feedback['content']['entity_analysis']['categories'].items():
        print(f"    {category}: {', '.join(entities)}")
    
    # Print key phrases
    if 'key_phrases' in feedback['content']['entity_analysis']:
        print("  Key Phrases:")
        # Split into groups of 5 for readability
        phrases = feedback['content']['entity_analysis']['key_phrases']
        for i in range(0, len(phrases), 5):
            print(f"    {', '.join(phrases[i:i+5])}")
    
    # Print statistics if available
    if 'statistics' in feedback['content']['entity_analysis']:
        stats = feedback['content']['entity_analysis']['statistics']
        print("  Statistics:")
        print(f"    Total unique entities: {stats.get('total_unique', 0)}")
        
        if 'most_mentioned' in stats:
            print("    Most mentioned entities:")
            for entity in stats['most_mentioned']:
                print(f"      {entity['entity']}: {entity['mentions']} mentions")
        
        if 'category_counts' in stats:
            print("    Entity counts by category:")
            for category, count in stats['category_counts'].items():
                print(f"      {category}: {count}")
    
    # Structure section
    print("\nStructure:")
    print(f"  Sentence count: {feedback['structure']['sentence_count']}")
    print(f"  Average sentence length: {feedback['structure']['avg_sentence_length']}")
    print(f"  Recommendation: {feedback['structure']['recommendation']}")
    
    # Language section
    print("\nLanguage:")
    print(f"  Average word length: {feedback['language']['avg_word_length']}")
    print(f"  Complex word ratio: {feedback['language']['complex_word_ratio']}%")
    print(f"  Recommendation: {feedback['language']['recommendation']}")
    
    # Overall score
    if 'overall_score' in feedback:
        print(f"\nOverall Score: {feedback['overall_score']}/10")
    
    # Suggestions section
    print("\nSuggestions for Improvement:")
    for i, suggestion in enumerate(feedback['suggestions'], 1):
        print(f"  {i}. {suggestion}")
    
    # Save the model
    evaluator.save_model("mun_speech_evaluator.joblib")
    
    # Later, you can load the model
    # loaded_evaluator = MUNSpeechEvaluator().load_model("mun_speech_evaluator.joblib")

if __name__ == "__main__":
    example_usage()