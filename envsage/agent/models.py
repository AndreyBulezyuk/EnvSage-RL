from django.db import models
from django.contrib.auth.models import User

class BaseModel(models.Model):
    id = models.AutoField(primary_key=True)
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='%(class)s_created')
    modified_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='%(class)s_modified')
    created_at = models.DateTimeField(auto_now_add=True)
    modified_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True

class Session(BaseModel):
    name = models.CharField(max_length=255, unique=True)
    description = models.TextField(blank=True, null=True)

    def __str__(self):
        return self.name

class Episode(BaseModel):
    session = models.ForeignKey(Session, on_delete=models.CASCADE, related_name='episodes')
    episode_index = models.IntegerField()
    mode = models.CharField(max_length=50)
    return_value = models.FloatField()
    length = models.IntegerField()
    terminated = models.BooleanField(default=False)
    truncated = models.BooleanField(default=False)
    summary_json = models.JSONField(blank=True, null=True)
    triggered_by_experiment = models.ForeignKey('Experiment', on_delete=models.SET_NULL, null=True, blank=True, related_name='triggered_episodes')
    triggered_by_hypothesis = models.ForeignKey('Hypothesis', on_delete=models.SET_NULL, null=True, blank=True, related_name='triggered_episodes')

    class Meta:
        unique_together = ('session', 'episode_index')

    def __str__(self):
        return f"Episode {self.episode_index} in {self.session.name}"

class Experiment(BaseModel):
    status_choices = [
        ('pending', 'Pending'), 
        ('running', 'Running'), 
        ('completed', 'Completed'), 
        # Experiments results were useful and implemented in the agent's policy
        ('implemented', 'Implemented'), 
        ('failed', 'Failed')
         ]

    session = models.ForeignKey(Session, on_delete=models.CASCADE, related_name='experiments')
    episode_index = models.IntegerField()
    name = models.CharField(max_length=150)
    description = models.TextField()
    target_question = models.JSONField(blank=True, null=True)
    reason = models.TextField(blank=True, null=True)
    result_summary_json = models.JSONField(blank=True, null=True)
    custom_action_python_oneline_method = models.CharField(max_length=3000, blank=True, null=True)
    status=models.CharField(max_length=20, default='pending', choices=status_choices)
    belongs_to_hypothesis = models.ForeignKey('Hypothesis', on_delete=models.SET_NULL, null=True, blank=True, related_name='experiments')

    class Meta:
        pass
        # unique_together = ('session', 'episode_index')

    def __str__(self):
        return f"Experiment {self.name} in {self.session.name}"

class Hypothesis(BaseModel):
    session = models.ForeignKey(Session, on_delete=models.CASCADE, related_name='hypotheses')
    episode_index = models.IntegerField()
    name = models.CharField(max_length=300)
    kind = models.CharField(max_length=80)
    statement = models.TextField()
    candidate_equation = models.TextField(blank=True, null=True)
    variables_json = models.JSONField(blank=True, null=True)
    confidence = models.FloatField(default=0.0)
    status = models.CharField(max_length=80)
    evidence = models.TextField()
    proposed_test = models.TextField()

    def __str__(self):
        return f"Hypothesis {self.name} in {self.session.name}"

class Interpretation(BaseModel):
    session = models.ForeignKey(Session, on_delete=models.CASCADE, related_name='interpretations')
    episode_index = models.IntegerField()
    feature_index = models.IntegerField()
    meanings_json = models.JSONField()
    confidence = models.FloatField(default=0.0)
    reason = models.TextField()

    def __str__(self):
        return f"Interpretation for feature {self.feature_index} in {self.session.name}"

class Constant(BaseModel):
    session = models.ForeignKey(Session, on_delete=models.CASCADE, related_name='constants')
    episode_index = models.IntegerField()
    name = models.CharField(max_length=200)
    value = models.CharField(max_length=200)
    confidence = models.FloatField(default=0.0)
    rationale = models.TextField()

    def __str__(self):
        return f"Constant {self.name} in {self.session.name}"

class LLMAudit(BaseModel):
    session = models.ForeignKey(Session, on_delete=models.CASCADE, related_name='llm_audits')
    episode_index = models.IntegerField(blank=True, null=True)
    purpose = models.CharField(max_length=100)
    prompt_preview = models.TextField()
    response_preview = models.TextField()

    def __str__(self):
        return f"LLM Audit for {self.purpose} in {self.session.name}"

class LLMCall(BaseModel):
    session = models.ForeignKey(Session, on_delete=models.CASCADE, related_name='llm_calls')
    episode_index = models.IntegerField(blank=True, null=True)
    purpose = models.CharField(max_length=100)
    prompt = models.TextField()
    response = models.TextField()
    model = models.CharField(max_length=100, default='gpt-5.1')

    def __str__(self):
        return f"LLM Call for {self.purpose} in {self.session.name}"
