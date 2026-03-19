from django.contrib import admin
from .models import Session, Episode, Experiment, Hypothesis, Interpretation, Constant, LLMAudit, LLMCall

class SessionAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'created_at', 'modified_at')
    search_fields = ('name',)
    ordering = ('-created_at',)

class EpisodeAdmin(admin.ModelAdmin):
    list_display = ('id', 'session', 'episode_index', 'mode', 'return_value', 'length', 'terminated', 'truncated', 'created_at')
    search_fields = ('session__name',)
    list_filter = ('mode', 'terminated', 'truncated')
    ordering = ('-created_at',)


class ExperimentAdmin(admin.ModelAdmin):
    list_display = ('id', 'session', 'episode_index', 'name', 'created_at')
    search_fields = ('session__name', 'name')
    ordering = ('-created_at',)

class HypothesisAdmin(admin.ModelAdmin):
    list_display = ('id', 'session', 'episode_index', 'name', 'kind', 'created_at')
    search_fields = ('session__name', 'name', 'kind')
    ordering = ('-created_at',)

class InterpretationAdmin(admin.ModelAdmin):
    list_display = ('id', 'session', 'episode_index', 'feature_index', 'confidence', 'created_at')
    search_fields = ('session__name',)
    list_filter = ('feature_index',)
    ordering = ('-created_at',)

class ConstantAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'value', 'created_at')
    search_fields = ('name',)
    ordering = ('-created_at',)

class LLMAuditAdmin(admin.ModelAdmin):
    list_display = ('id', 'session', 'episode_index', 'purpose', 'prompt_preview', 'response_preview', 'created_at')
    search_fields = ('session__name', 'purpose')
    ordering = ('-created_at',)

class LLMCallAdmin(admin.ModelAdmin):
    list_display = ('id', 'session', 'episode_index', 'model', 'purpose', 'prompt', 'response', 'created_at')
    search_fields = ('session__name', 'model', 'purpose')
    ordering = ('-created_at',)

admin.site.register(Session, SessionAdmin)
admin.site.register(Episode, EpisodeAdmin)
admin.site.register(Experiment, ExperimentAdmin)
admin.site.register(Hypothesis, HypothesisAdmin)
admin.site.register(Interpretation, InterpretationAdmin)
admin.site.register(Constant, ConstantAdmin)
admin.site.register(LLMAudit, LLMAuditAdmin)
admin.site.register(LLMCall, LLMCallAdmin)